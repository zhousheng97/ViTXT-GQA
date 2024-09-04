# Copyright (c) Facebook, Inc. and its affiliates.
from pythia.common.registry import Registry
from pythia.datasets.videoqa.gt_box_clipocr.dataset import  GTBOX
from pythia.datasets.vqa.textvqa.builder import TextVQABuilder


@Registry.register_builder("gt_box")
class PipelineTextVQABuilder(TextVQABuilder):
    def __init__(self):
        super().__init__()
        self.dataset_name = "gt_box"
        self.set_dataset_class(GTBOX)
