# Copyright (c) Facebook, Inc. and its affiliates.
"""
The metrics module contains implementations of various metrics used commonly to
understand how well our models are performing. For e.g. accuracy, vqa_accuracy,
r@1 etc.

For implementing your own metric, you need to follow these steps:

1. Create your own metric class and inherit ``BaseMetric`` class.
2. In the ``__init__`` function of your class, make sure to call
   ``super().__init__('name')`` where 'name' is the name of your metric. If
   you require any parameters in your ``__init__`` function, you can use
   keyword arguments to represent them and metric constructor will take care of
   providing them to your class from config.
3. Implement a ``calculate`` function which takes in ``SampleList`` and
   `model_output` as input and return back a float tensor/number.
4. Register your metric with a key 'name' by using decorator,
   ``@registry.register_metric('name')``.

Example::

    import torch

    from pythia.common.registry import registry
    from pythia.modules.metrics import BaseMetric

    @registry.register_metric("some")
    class SomeMetric(BaseMetric):
        def __init__(self, some_param=None):
            super().__init__("some")
            ....

        def calculate(self, sample_list, model_output):
            metric = torch.tensor(2, dtype=torch.float)
            return metric

Example config for above metric::

    model_attributes:
        pythia:
            metrics:
            - type: some
              params:
                some_param: a
"""

import collections
import torch
import numpy as np
from pythia.common.registry import registry


class Metrics:
    """Internally used by Pythia, Metrics acts as wrapper for handling
    calculation of metrics over various metrics specified by the model in
    the config. It initializes all of the metrics and when called it runs
    calculate on each of them one by one and returns back a dict with proper
    naming back. For e.g. an example dict returned by Metrics class:
    ``{'val/vqa_accuracy': 0.3, 'val/r@1': 0.8}``

    Args:
        metric_list (List[ConfigNode]): List of ConfigNodes where each ConfigNode
                                        specifies name and parameters of the
                                        metrics used.
    """

    def __init__(self, metric_list):
        if not isinstance(metric_list, list):
            metric_list = [metric_list]

        self.writer = registry.get("writer")
        self.metrics = self._init_metrics(metric_list)

    def _init_metrics(self, metric_list):
        metrics = {}
        for metric in metric_list:
            params = {}
            if isinstance(metric, collections.abc.Mapping):
                if not hasattr(metric, "type"):
                    raise ValueError(
                        "Metric {} needs to have 'type' attribute".format(metric)
                    )
                metric = metric.type
                params = getattr(metric, "params", {})
            else:
                if not isinstance(metric, str):
                    raise TypeError(
                        "Metric {} has inappropriate type"
                        "'dict' or 'str' allowed".format(metric)
                    )

            metric_cls = registry.get_metric_class(metric)
            if metric_cls is None:
                raise ValueError(
                    "No metric named {} registered to registry".format(metric)
                )
            metrics[metric] = metric_cls(**params)

        return metrics

    def __call__(self, sample_list, model_output, *args, **kwargs):
        values = {}
        if not hasattr(sample_list, "targets"):
            return values

        dataset_type = sample_list.dataset_type
        dataset_name = sample_list.dataset_name

        with torch.no_grad():
            if dataset_type == "train":
                self.metrics = {key: value for key, value in self.metrics.items() if key in {'textvqa_accuracy', 'stvqa_anls'}}

            for metric_name, metric_object in self.metrics.items():
                key = "{}/{}/{}".format(dataset_type, dataset_name, metric_name)
                values[key] = metric_object._calculate_with_checks(
                    sample_list, model_output, *args, **kwargs
                )

                if not isinstance(values[key], torch.Tensor):
                    values[key] = torch.tensor(values[key], dtype=torch.float)
                else:
                    values[key] = values[key].float()

                if values[key].dim() == 0:
                    values[key] = values[key].view(1)

        registry.register(
            "{}.{}.{}".format("metrics", sample_list.dataset_name, dataset_type), values
        )

        return values


class BaseMetric:
    """Base class to be inherited by all metrics registered to Pythia. See
    the description on top of the file for more information. Child class must
    implement ``calculate`` function.

    Args:
        name (str): Name of the metric.

    """

    def __init__(self, name, *args, **kwargs):
        self.name = name

    def calculate(self, sample_list, model_output, *args, **kwargs):
        """Abstract method to be implemented by the child class. Takes
        in a ``SampleList`` and a dict returned by model as output and
        returns back a float tensor/number indicating value for this metric.

        Args:
            sample_list (SampleList): SampleList provided by the dataloader for the
                                current iteration.
            model_output (Dict): Output dict from the model for the current
                                 SampleList

        Returns:
            torch.Tensor|float: Value of the metric.

        """
        # Override in your child class
        raise NotImplementedError(
            "'calculate' must be implemented in the child class"
        )

    def __call__(self, *args, **kwargs):
        return self.calculate(*args, **kwargs)

    def _calculate_with_checks(self, *args, **kwargs):
        value = self.calculate(*args, **kwargs)
        return value


@registry.register_metric("textvqa_accuracy")
class TextVQAAccuracy(BaseMetric):
    def __init__(self):
        super().__init__("textvqa_accuracy")
        import pythia.utils.m4c_evaluators as evaluators
        self.evaluator = evaluators.TextVQAAccuracyEvaluator()

    def calculate(self, sample_list, model_output, *args, **kwargs):
        answer_processor = registry.get(
            sample_list.dataset_name + "_answer_processor"
        )
        batch_size = sample_list.context_tokens_enc.size(0)
        pred_answers = model_output["pos_scores"].argmax(dim=-1)
        context_tokens_enc = sample_list.context_tokens_enc.cpu().numpy()
        gt_answers_enc = sample_list.gt_answers_enc.cpu().numpy()
        answer_space_size = answer_processor.get_true_vocab_size()

        predictions = []
        from pythia.utils.objects_to_byte_tensor import dec_bytes2obj
        from pythia.utils.text_utils import word_tokenize
        for idx in range(batch_size):
            context_tokens = dec_bytes2obj(context_tokens_enc[idx])
            answer_words = []
            for answer_id in pred_answers[idx].tolist():
                if answer_id >= answer_space_size:
                    answer_id -= answer_space_size
                    answer_words.append(
                        word_tokenize(context_tokens[answer_id])
                    )
                else:
                    if answer_id == answer_processor.EOS_IDX:
                        break
                    answer_words.append(
                        answer_processor.answer_vocab.idx2word(answer_id)
                    )

            pred_answer = ' '.join(answer_words).replace(" 's", "'s")
            gt_answers = dec_bytes2obj(gt_answers_enc[idx])
            predictions.append({
                "pred_answer": pred_answer,
                "gt_answers": gt_answers,
            })
        pred_scores = []
        _, accuracy = self.evaluator.eval_pred_list(pred_scores, predictions)
        accuracy = torch.tensor(accuracy).to(sample_list.context_tokens_enc.device) 

        return accuracy


@registry.register_metric("stvqa_anls")
class STVQAANLS(TextVQAAccuracy):
    def __init__(self):
        self.name = "stvqa_anls"
        import pythia.utils.m4c_evaluators as evaluators
        self.evaluator = evaluators.STVQAANLSEvaluator()


    
@registry.register_metric("IOU@0.3")
class BoxGroundAccuracy(BaseMetric):
    '''
    the IoU of the grounded bbox and the labeled bbox are greater than 0.3 indicates that the location is correct
    '''
    def __init__(self):
        super().__init__("IOU@0.3")
        import pythia.utils.m4c_evaluators as evaluators
        self.evaluator = evaluators.BoxGroundAccuracyEvaluator()

    def find_dict_by_id(self, dict_list, target_id):
        for dictionary in dict_list:
            if "question_id" in dictionary and dictionary["question_id"] == target_id:
                return dictionary
        return None
    
    def calculate(self, sample_list, model_output, *args, **kwargs):
        if sample_list['dataset_type'] == 'val':
            self.ground_info_dir = '/data/zsheng/Data_T5_ViteVQA/data/m4vitevqa/ground_annotation/grouding_anno_t1s2val.npy'
        else:
            self.ground_info_dir = '/data/zsheng/Data_T5_ViteVQA/data/m4vitevqa/ground_annotation/grouding_anno_t1s2test.npy'
        self.ground_info = np.load(self.ground_info_dir, allow_pickle=True)[1:]

        batch_size = sample_list.frame_num.size(0)
        pred_frames = model_output["ground_frame"].detach().to('cpu').numpy().tolist()
        pred_boxs = model_output["ground_box"].detach().to('cpu').tolist()
        frame_topk = int(model_output["frame_topk"].detach().to('cpu'))
        ocr_topk = int(model_output["ocr_topk"].detach().to('cpu'))
  
        predictions = []
        for idx in range(batch_size):
            gt_info = self.find_dict_by_id(self.ground_info, sample_list['question_id'][idx])
            st_gt = gt_info['spatial_temporal_gt']
            video_fps = gt_info['fps']
            width, height = gt_info['width'], gt_info['height']
            pred_box = pred_boxs[idx] 
            pred_frame = pred_frames[idx] 
            predictions.append({
                "pred_frame": pred_frame,
                "pred_box": pred_box,
                "frame_topk": frame_topk,
                "ocr_topk": ocr_topk,
                "st_gt": st_gt,
                "video_fps": video_fps,
                "width": width,
                "height": height,
            })
        pred_scores = []
        _, box_accuracy = self.evaluator.eval_pred_list(pred_scores, predictions, threshold=0.3)
        box_accuracy = torch.tensor(box_accuracy).to(sample_list.frame_num.device)  # .cuda()

        return box_accuracy

@registry.register_metric("IOU@0.5")
class BoxGroundAccuracy(BaseMetric):
    '''
    the IoU of the grounded bbox and the labeled bbox are greater than 0.5 indicates that the location is correct
    '''
    def __init__(self):
        super().__init__("IOU@0.5")
        import pythia.utils.m4c_evaluators as evaluators
        self.evaluator = evaluators.BoxGroundAccuracyEvaluator()

    def find_dict_by_id(self, dict_list, target_id):
        for dictionary in dict_list:
            if "question_id" in dictionary and dictionary["question_id"] == target_id:
                return dictionary
        return None
    
    def calculate(self, sample_list, model_output, *args, **kwargs):
        if sample_list['dataset_type'] == 'val':
            self.ground_info_dir = '/data/zsheng/Data_T5_ViteVQA/data/m4vitevqa/ground_annotation/grouding_anno_t1s2val.npy'
        else:
            self.ground_info_dir = '/data/zsheng/Data_T5_ViteVQA/data/m4vitevqa/ground_annotation/grouding_anno_t1s2test.npy'
        self.ground_info = np.load(self.ground_info_dir, allow_pickle=True)[1:]

        batch_size = sample_list.frame_num.size(0)
        pred_frames = model_output["ground_frame"].detach().to('cpu').numpy().tolist()
        pred_boxs = model_output["ground_box"].detach().to('cpu').tolist()
        frame_topk = int(model_output["frame_topk"].detach().to('cpu'))
        ocr_topk = int(model_output["ocr_topk"].detach().to('cpu'))
  
        predictions = []
        for idx in range(batch_size):
            gt_info = self.find_dict_by_id(self.ground_info, sample_list['question_id'][idx])
            st_gt = gt_info['spatial_temporal_gt']
            video_fps = gt_info['fps']
            width, height = gt_info['width'], gt_info['height']
            pred_box = pred_boxs[idx] 
            pred_frame = pred_frames[idx] 
            predictions.append({
                "pred_frame": pred_frame,
                "pred_box": pred_box,
                "frame_topk": frame_topk,
                "ocr_topk": ocr_topk,
                "st_gt": st_gt,
                "video_fps": video_fps,
                "width": width,
                "height": height,
            })

        pred_scores = []
        _, box_accuracy = self.evaluator.eval_pred_list(pred_scores, predictions, threshold=0.5)
        box_accuracy = torch.tensor(box_accuracy).to(sample_list.frame_num.device)  # .cuda()

        return box_accuracy
    

@registry.register_metric("GQA@0.5")
class BoxGroundAccuracy(BaseMetric):
    '''
    the IoU of the grounded bbox and the labeled bbox are greater than 0.5 indicates that the location is correct
    '''
    def __init__(self):
        super().__init__("GQA@0.5")
        import pythia.utils.m4c_evaluators as evaluators
        self.box_evaluator = evaluators.BoxGroundAccuracyEvaluator()
        self.qa_evaluator = evaluators.TextVQAAccuracyEvaluator()


    def find_dict_by_id(self, dict_list, target_id):
        for dictionary in dict_list:
            if "question_id" in dictionary and dictionary["question_id"] == target_id:
                return dictionary
        return None
    
    def calculate(self, sample_list, model_output, *args, **kwargs):
        if sample_list['dataset_type'] == 'val':
            self.ground_info_dir = '/data/zsheng/Data_T5_ViteVQA/data/m4vitevqa/ground_annotation/grouding_anno_t1s2val.npy'
        else:
            self.ground_info_dir = '/data/zsheng/Data_T5_ViteVQA/data/m4vitevqa/ground_annotation/grouding_anno_t1s2test.npy'
        self.ground_info = np.load(self.ground_info_dir, allow_pickle=True)[1:]

        batch_size = sample_list.frame_num.size(0)
        pred_frames = model_output["ground_frame"].detach().to('cpu').numpy().tolist()
        pred_boxs = model_output["ground_box"].detach().to('cpu').tolist()
        frame_topk = int(model_output["frame_topk"].detach().to('cpu'))
        ocr_topk = int(model_output["ocr_topk"].detach().to('cpu'))

        # Box grounding
        box_predictions = []
        for idx in range(batch_size):
            gt_info = self.find_dict_by_id(self.ground_info, sample_list['question_id'][idx])
            st_gt = gt_info['spatial_temporal_gt']
            video_fps = gt_info['fps']
            width, height = gt_info['width'], gt_info['height']
            pred_box = pred_boxs[idx] 
            pred_frame = pred_frames[idx] 
            box_predictions.append({
                "pred_frame": pred_frame,
                "pred_box": pred_box,
                "frame_topk": frame_topk,
                "ocr_topk": ocr_topk,
                "st_gt": st_gt,
                "video_fps": video_fps,
                "width": width,
                "height": height,
            })
        box_pred_scores = []
        box_pred_scores, _ = self.box_evaluator.eval_pred_list(box_pred_scores, box_predictions, threshold=0.5)

        # QA 
        answer_processor = registry.get(
            sample_list.dataset_name + "_answer_processor"
        )
        batch_size = sample_list.context_tokens_enc.size(0)
        pred_answers = model_output["pos_scores"].argmax(dim=-1)
        context_tokens_enc = sample_list.context_tokens_enc.cpu().numpy()
        gt_answers_enc = sample_list.gt_answers_enc.cpu().numpy()
        answer_space_size = answer_processor.get_true_vocab_size()

        qa_predictions = []
        from pythia.utils.objects_to_byte_tensor import dec_bytes2obj
        from pythia.utils.text_utils import word_tokenize
        for idx in range(batch_size):
            context_tokens = dec_bytes2obj(context_tokens_enc[idx])
            answer_words = []
            for answer_id in pred_answers[idx].tolist():
                if answer_id >= answer_space_size:
                    answer_id -= answer_space_size
                    answer_words.append(
                        word_tokenize(context_tokens[answer_id])
                    )
                else:
                    if answer_id == answer_processor.EOS_IDX:
                        break
                    answer_words.append(
                        answer_processor.answer_vocab.idx2word(answer_id)
                    )

            pred_answer = ' '.join(answer_words).replace(" 's", "'s")
            gt_answers = dec_bytes2obj(gt_answers_enc[idx])
            qa_predictions.append({
                "pred_answer": pred_answer,
                "gt_answers": gt_answers,
            })
        qa_pred_scores = []
        qa_pred_scores, _ = self.qa_evaluator.eval_pred_list(qa_pred_scores, qa_predictions)

        gqa_score = []
        for i in range(batch_size):
            if box_pred_scores[i] == 1 and  qa_pred_scores[i] == 1:
                gqa_score.append(1)
            else:
                gqa_score.append(0)

        gqa_accuracy = sum(gqa_score) / len(gqa_score)
        gqa_accuracy = torch.tensor(gqa_accuracy).to(sample_list.frame_num.device)  # .cuda()

        return gqa_accuracy

@registry.register_metric("GQA@0.3")
class BoxGroundAccuracy(BaseMetric):
    '''
    the IoU of the grounded bbox and the labeled bbox are greater than 0.5 indicates that the location is correct
    '''
    def __init__(self):
        super().__init__("GQA@0.3")
        import pythia.utils.m4c_evaluators as evaluators
        self.box_evaluator = evaluators.BoxGroundAccuracyEvaluator()
        self.qa_evaluator = evaluators.TextVQAAccuracyEvaluator()


    def find_dict_by_id(self, dict_list, target_id):
        for dictionary in dict_list:
            if "question_id" in dictionary and dictionary["question_id"] == target_id:
                return dictionary
        return None
    
    def calculate(self, sample_list, model_output, *args, **kwargs):
        if sample_list['dataset_type'] == 'val':
            self.ground_info_dir = '/data/zsheng/Data_T5_ViteVQA/data/m4vitevqa/ground_annotation/grouding_anno_t1s2val.npy'
        else:
            self.ground_info_dir = '/data/zsheng/Data_T5_ViteVQA/data/m4vitevqa/ground_annotation/grouding_anno_t1s2test.npy'
        self.ground_info = np.load(self.ground_info_dir, allow_pickle=True)[1:]

        batch_size = sample_list.frame_num.size(0)
        pred_frames = model_output["ground_frame"].detach().to('cpu').numpy().tolist()
        pred_boxs = model_output["ground_box"].detach().to('cpu').tolist()
        frame_topk = int(model_output["frame_topk"].detach().to('cpu'))
        ocr_topk = int(model_output["ocr_topk"].detach().to('cpu'))

        # Box grounding
        box_predictions = []
        for idx in range(batch_size):
            gt_info = self.find_dict_by_id(self.ground_info, sample_list['question_id'][idx])
            st_gt = gt_info['spatial_temporal_gt']
            video_fps = gt_info['fps']
            width, height = gt_info['width'], gt_info['height']
            pred_box = pred_boxs[idx] 
            pred_frame = pred_frames[idx] 
            box_predictions.append({
                "pred_frame": pred_frame,
                "pred_box": pred_box,
                "frame_topk": frame_topk,
                "ocr_topk": ocr_topk,
                "st_gt": st_gt,
                "video_fps": video_fps,
                "width": width,
                "height": height,
            })
        box_pred_scores = []
        box_pred_scores, _ = self.box_evaluator.eval_pred_list(box_pred_scores, box_predictions, threshold=0.3)

        # QA 
        answer_processor = registry.get(
            sample_list.dataset_name + "_answer_processor"
        )
        batch_size = sample_list.context_tokens_enc.size(0)
        pred_answers = model_output["pos_scores"].argmax(dim=-1)
        context_tokens_enc = sample_list.context_tokens_enc.cpu().numpy()
        gt_answers_enc = sample_list.gt_answers_enc.cpu().numpy()
        answer_space_size = answer_processor.get_true_vocab_size()

        qa_predictions = []
        from pythia.utils.objects_to_byte_tensor import dec_bytes2obj
        from pythia.utils.text_utils import word_tokenize
        for idx in range(batch_size):
            context_tokens = dec_bytes2obj(context_tokens_enc[idx])
            answer_words = []
            for answer_id in pred_answers[idx].tolist():
                if answer_id >= answer_space_size:
                    answer_id -= answer_space_size
                    answer_words.append(
                        word_tokenize(context_tokens[answer_id])
                    )
                else:
                    if answer_id == answer_processor.EOS_IDX:
                        break
                    answer_words.append(
                        answer_processor.answer_vocab.idx2word(answer_id)
                    )

            pred_answer = ' '.join(answer_words).replace(" 's", "'s")
            gt_answers = dec_bytes2obj(gt_answers_enc[idx])
            qa_predictions.append({
                "pred_answer": pred_answer,
                "gt_answers": gt_answers,
            })
        qa_pred_scores = []
        qa_pred_scores, _ = self.qa_evaluator.eval_pred_list(qa_pred_scores, qa_predictions)

        gqa_score = []
        for i in range(batch_size):
            if box_pred_scores[i] == 1 and  qa_pred_scores[i] == 1:
                gqa_score.append(1)
            else:
                gqa_score.append(0)

        gqa_accuracy = sum(gqa_score) / len(gqa_score)
        gqa_accuracy = torch.tensor(gqa_accuracy).to(sample_list.frame_num.device)  # .cuda()

        return gqa_accuracy
    
