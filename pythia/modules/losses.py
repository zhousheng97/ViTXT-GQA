# Copyright (c) Facebook, Inc. and its affiliates.
"""
Losses module contains implementations for various losses used generally
in vision and language space. One can register custom losses to be detected by
pythia using the following example.

.. code::

   from pythia.common.registry import registry
   from torch import nn


   @registry.register_loss("custom")
   class CustomLoss(nn.Module):
       ...

Then in your model's config you can specify ``losses`` attribute to use this loss
in the following way:

.. code::

   model_attributes:
       some_model:
           losses:
               - type: custom
               - params: {}
"""
import warnings
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
from torch.nn.utils.rnn import pack_padded_sequence

from pythia.common.registry import registry

from pythia.utils.compute_anls import ANLSEvaluator
import os

class Losses(nn.Module):
    """``Losses`` acts as an abstraction for instantiating and calculating
    losses. ``BaseModel`` instantiates this class based on the `losses`
    attribute in the model's configuration `model_attributes`. ``loss_list``
    needs to be a list for each separate loss containing `type` and `params`
    attributes.

    Args:
        loss_list (List[ConfigNode]): Description of parameter `loss_list`.

    Example::

        # losses:
        # - type: logit_bce
        # Can also contain `params` to specify that particular loss's init params
        # - type: combined
        config = [{"type": "logit_bce"}, {"type": "combined"}]
        losses = Losses(config)

    .. note::

        Since, ``Losses`` is instantiated in the ``BaseModel``, normal end user
        mostly doesn't need to use this class.

    Attributes:
        losses: List containing instanttions of each loss
                                   passed in config
    """

    def __init__(self, loss_list):
        super().__init__()
        self.losses = []
        tp = registry.get("config").training_parameters
        self._evalai_inference = tp.evalai_inference
        for loss in loss_list:
            self.losses.append(PythiaLoss(loss))

    def forward(self, sample_list, model_output, *args, **kwargs):
        """Takes in the original ``SampleList`` returned from DataLoader
        and `model_output` returned from the model and returned a Dict containing
        loss for each of the losses in `losses`.

        Args:
            sample_list (SampleList): SampleList given be the dataloader.
            model_output (Dict): Dict returned from model as output.

        Returns:
            Dict: Dictionary containing loss value for each of the loss.

        """
        output = {}
        if not hasattr(sample_list, "targets"):
            if not self._evalai_inference:
                warnings.warn(
                    "Sample list has not field 'targets', are you "
                    "sure that your ImDB has labels? you may have "
                    "wanted to run with --evalai_inference 1"
                )
            return output

        for loss in self.losses:
            output.update(loss(sample_list, model_output, *args, **kwargs))


        registry_loss_key = "{}.{}.{}".format(
            "losses", sample_list.dataset_name, sample_list.dataset_type
        )
        # Register the losses to registry
        registry.register(registry_loss_key, output)

        return output


class PythiaLoss(nn.Module):
    """Internal Pythia helper and wrapper class for all Loss classes.
    It makes sure that the value returned from a Loss class is a dict and
    contain proper dataset type in keys, so that it is easy to figure out
    which one is the val loss and which one is train loss.

    For example: it will return ``{"val/logit_bce": 27.4}``, in case
    `logit_bce` is used and SampleList is from `val` set.

    Args:
        params (type): Description of parameter `params`.

    .. note::

        Since, ``PythiaLoss`` is used by the ``Losses`` class, end user
        doesn't need to worry about it.
    """

    def __init__(self, params={}):
        super().__init__()
        self.writer = registry.get("writer")
        if "type" not in params:
            raise ValueError(
                "Parameters to loss must have 'type' field to"
                "specify type of loss to instantiate"
            )

        loss_name = params["type"]
        self.name = loss_name
        self.weight = params['weight']

        loss_class = registry.get_loss_class(loss_name)

        if loss_class is None:
            raise ValueError(
                "No loss named {} is registered to registry".format(loss_name)
            )
        # Special case of multi as it requires an array
        if loss_name == "multi":
            self.loss_criterion = loss_class(params)
        else:
            loss_params = params.get("params", {})
            self.loss_criterion = loss_class(**loss_params)

    def forward(self, sample_list, model_output, *args, **kwargs):
        loss = self.loss_criterion(sample_list, model_output, *args, **kwargs)
        # weighted loss
        loss = self.weight * loss

        if not isinstance(loss, torch.Tensor):
            loss = torch.tensor(loss, dtype=torch.float)

        if loss.dim() == 0:
            loss = loss.view(1)

        key = "{}/{}/{}".format(
            sample_list.dataset_type, sample_list.dataset_name, self.name
        )

        return {key: loss}


@registry.register_loss("bce")
class BinaryCrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, sample_list, model_output):
        """Calculates and returns the binary cross entropy.

        Args:
            sample_list (SampleList): SampleList containing `targets` attribute.
            model_output (Dict): Model output containing `scores` attribute.

        Returns:
            torch.FloatTensor: Float value for loss.

        """
        scores = model_output["scores"]
        targets = sample_list["targets"]
        loss = F.binary_cross_entropy(scores, targets, reduction="mean")

        return loss * targets.size(1)



def kl_div(log_x, y):
    y_is_0 = torch.eq(y.data, 0)
    y.data.masked_fill_(y_is_0, 1)
    log_y = torch.log(y)
    y.data.masked_fill_(y_is_0, 0)
    res = y * (log_y - log_x)

    return torch.sum(res, dim=1, keepdim=True)


@registry.register_loss("multi")
class MultiLoss(nn.Module):
    """A loss for combining multiple losses with weights.

    Args:
        params (List(Dict)): A list containing parameters for each different loss
                             and their weights.

    Example::

        # MultiLoss works with config like below where each loss's params and
        # weights are defined
        losses:
        - type: multi
          params:
          - type: logit_bce
            weight: 0.3
            params: {}
          - type: attention_supervision
            weight: 0.7
            params: {}

    """

    def __init__(self, params):
        super().__init__()
        self.losses = []
        self.losses_weights = []
        self.writer = registry.get("writer")

        self.loss_names = []

        for loss_params in params["params"]:
            self.loss_names.append(loss_params["type"])
            loss_fn = PythiaLoss(loss_params)
            loss_weight = loss_params.get("weight", {})
            self.losses.append(loss_fn)
            self.losses_weights.append(loss_weight)

    def forward(self, sample_list, model_output, *args, **kwargs):
        """Calculates and returns the multi loss.

        Args:
            sample_list (SampleList): SampleList containing `attentions` attribute.
            model_output (Dict): Model output containing `attention_supervision`
                                 attribute.

        Returns:
            torch.FloatTensor: Float value for loss.

        """
        loss = 0
        for idx, loss_fn in enumerate(self.losses):
            loss_dict = loss_fn(sample_list, model_output, *args, **kwargs)
            for key, value in loss_dict.items():
                loss += self.losses_weights[idx] * value
        return loss




@registry.register_loss("bce_kl_combined")
class CombinedLoss(nn.Module):
    def __init__(self, weight_softmax):
        super().__init__()
        self.weight_softmax = weight_softmax

    def forward(self, sample_list, model_output):
        pred_score = model_output["scores"]
        target_score = sample_list["targets"]

        tar_sum = torch.sum(target_score, dim=1, keepdim=True)
        tar_sum_is_0 = torch.eq(tar_sum, 0)
        tar_sum.masked_fill_(tar_sum_is_0, 1.0e-06)
        tar = target_score / tar_sum

        res = F.log_softmax(pred_score, dim=1)
        loss1 = kl_div(res, tar)
        loss1 = torch.sum(loss1) / loss1.size(0)

        loss2 = F.binary_cross_entropy_with_logits(
            pred_score, target_score, reduction="mean"
        )
        loss2 *= target_score.size(1)

        loss = self.weight_softmax * loss1 + loss2

        return loss


@registry.register_loss("bce_loss")
class BCEWithMaskLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.one = torch.Tensor([1.])

    def forward(self, sample_list, model_output):
        scores = model_output["scores"]
        targets = sample_list["targets"]
        loss_mask = sample_list["train_loss_mask"]
        
        assert scores.dim() == 3 and loss_mask.dim() == 2

        losses = F.binary_cross_entropy_with_logits( 
            scores, targets, reduction="none"
        )  # [bs, 12, 1024]
        losses *= loss_mask.unsqueeze(-1)

        count = torch.max(torch.sum(loss_mask), self.one.to(losses.device))
        loss = torch.sum(losses) / count
        return loss  


@registry.register_loss("pos_bce_loss")
class POSBCEWithMaskLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.one = torch.Tensor([1.])

    def forward(self, sample_list, model_output):
        scores = model_output["pos_scores"]
        targets = sample_list["targets"]
        loss_mask = sample_list["train_loss_mask"]
        
        assert scores.dim() == 3 and loss_mask.dim() == 2

        losses = F.binary_cross_entropy_with_logits( 
            scores, targets, reduction="none"
        )  # [bs, 12, 1024]
        losses *= loss_mask.unsqueeze(-1)

        count = torch.max(torch.sum(loss_mask), self.one.to(losses.device))
        loss = torch.sum(losses) / count
        return loss  
    
    
@registry.register_loss("InfoNCE")
class InfoNCE(nn.Module):
    def __init__(self, temperature=0.1, reduction='mean', negative_mode='unpaired'):
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction
        self.negative_mode = negative_mode

    def transpose(self, x):
        return x.transpose(-2, -1)

    def normalize(self, *xs):
        return [None if x is None else F.normalize(x, dim=-1) for x in xs]


    def forward(self, sample_list, model_output, temperature=0.1, reduction='mean', negative_mode='paired'):
        query = model_output['ref_scores']
        positive_key = model_output['pos_scores']
        negative_key = model_output['neg_scores']

        # Normalize to unit vectors
        query, positive_key, negative_key = self.normalize(query, positive_key, negative_key) # [bs, 12, 5964]
        
        # calculate cosine similarity between the anchor and the positive answer
        batch_size = query.size(0)
        query = query.view(batch_size, -1)
        positive_key = positive_key.view(batch_size, -1)
        positive_logit  = F.cosine_similarity(query, positive_key, dim=1).unsqueeze(1)
    
        # calculate cosine similarity between the anchor and the negative answer
        negative_key = negative_key.view(batch_size, -1)
        negative_logit  = F.cosine_similarity(query, negative_key, dim=1).unsqueeze(1)

        # First index in last dimension are the positive samples
        logits = torch.cat([positive_logit, negative_logit], dim=1) # [bs, 2]
        labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)  #  index 0 denotes the first element is positive

        loss = F.cross_entropy(logits / temperature, labels, reduction=reduction) # [bs, 5964]

        return loss

  