# Copyright (c) Facebook, Inc. and its affiliates.
from pythia.common.registry import Registry
from pythia.datasets.videoqa.vtextgqa.dataset import  VTEXTGQADataset
from pythia.datasets.vqa.textvqa.builder import TextVQABuilder


@Registry.register_builder("vtextgqa")
class VTEXTGQABuilder(TextVQABuilder):
    def __init__(self):
        super().__init__()
        self.dataset_name = "vtextgqa"
        self.set_dataset_class(VTEXTGQADataset)
