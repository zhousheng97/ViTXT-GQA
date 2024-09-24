# Scene-Text Grounding for Text-Based Video Question Answering

[Paper](https://example.com)


## Introduction
In this work, we propose a novel **Grounded TextVideoQA** task by forcing the models to answer the questions and spatio-temporally localize the relevant scene texts, thus promoting a research trend towards interpretable QA. The task not only encourages visual evidence for answer predictions, but also isolates the challenges inherited in QA and scene text recognition, enabling the diagnosis of the root causes for failure predictions, 𝑒.𝑔., wrong QA or wrong scene text recognition? To achieve grounded TextVideoQA, we propose a baseline **Temporal to-Spatial (T2S)** model. The model highlights a disentangled temporal- and spatial-contrastive learning strategy for weakly grounding and grounded QA. Finally, to evaluate grounded TextVideoQA, we construct a new dataset **VText-GQA**, by extending the existing largest TextVideoQA dataset with answer grounding (spatio-temporal location) labels. 

This repository provides the code for our paper, including:

- Temporal to-Spatial (T2S) baseline model and VText-GQA benchmark dataset.
- Data preprocessing instructions, including data preprocessing and feature extraction scripts, as well as preprocessed features.
- Training and evaluation scripts and checkpoints.


## Installation
Clone this repository, and build it with the following command.

```
conda create -n vtextgqa python==3.8
conda activate vtextgqa
conda install pytorch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 pytorch-cuda=12.1 -c pytorch -c nvidia

git clone https://github.com/zhousheng97/T2S.git
cd T2S
pip install -r requirements.txt
python setup.py build develop
```

## Data Preparation
Please create a data folder ```root/data/``` outside this repo folder ```root/T2S/``` so that the two folders are in the same directory.

* **Raw Video and Video Feature**.
You can directly download the provided video feature [video feature path](https://drive.google.com/file/d/1v7_0eiGtoDMt5kwz1jsPZ25Bo_8hXNjj/view?usp=drive_link) or apply [here](https://github.com/bytedance/VTVQA) to download the raw video and then extract features. If you download the raw videos, you need to decode each video at 10fps and then extract the frame feature of ViT via the script provided in ```T2S/tools/video_feat/obtain_vit_feat.py```. Extract video feature into ```data/fps10_video_vit_feat```.

* **OCR Detection and Recognition**.
Based on the OCR detector [TransVTSpotter](https://github.com/weijiawu/TransVTSpotter), we provide the recognition results of OCR recognition systems [ABINet](https://github.com/FangShancheng/ABINet) and [CLIPOCR](https://github.com/wzx99/CLIPOCR), the download links are: 
[vtextgqa_abinet](https://drive.google.com/file/d/1MNgnMgON38iiWbKGMwFVKtuQiorC4UyG/view?usp=drive_link) and [vtextgqa_clip](https://drive.google.com/file/d/1h3L9CN_Z0ihrmKsNruXf3UnF2rjulXik/view?usp=drive_link).

* **Dataset Annotation**.
We provide the dataset files [here](https://drive.google.com/drive/folders/1JOOifZJOk6pvqHE2MDjpyVi4BcahfKge?usp=drive_link), including grounding annotation files, QA files, and vocabulary files.

* **Other**. The fixed vocabulary is obtained by ```T2S/pythia/scripts/extract_vocabulary.py```

Repo structure as below:
```
root
├── T2S
├── data
│   └── fps10_ocr_detection
│   └── fps10_ocr_detection_ClipOCR
│   └── fps10_video_vit_feat
│   └── vtextgqa
│       ├── ground_annotation
│       ├── qa_annotation
│       ├── vocabulary
```

The following is an example of a spatio-temporal grounding label  file：
```
[{
      "question_id": 12393,
      "video_id": "02669",
      "fps": 10.0,
      "frames": 61,
      "duration": 6.1,
      "height": 1080,
      "width": 1920,
      "spatial_temporal_gt": [
      {
          "temporal_gt": [
              5.1,
              5.2
          ],
          "bbox_gt": {
              "51": [
                  799.2452830188679,
                  271.6981132075472,
                  858.1132075471698,
                  326.0377358490566
              ],
              "52": [
                  881.5686274509803,
                  295.1162790697674,
                  928.6274509803922,
                  357.906976744186
              ]
          }
      }]
  }]
```

## Training and Evaluation
The training and evaluation commands can be found in the ```T2S/scripts```. The config files can be found in the ```T2S/configs```.

* Train the model on the training set:
```
# bash scripts/<train.sh> <GPU_ids> <save_dir>

bash scripts/train_t2s_abinet.sh 0,1 vtextgqa_debug_abinet
```

* Evaluate the pretrained model on the validation/test sets:
```
# bash scripts/<val.sh> <GPU_ids> <save_dir> <checkpoint> <run_type>

bash scripts/val_t2s_abinet.sh 0,1 vtextgqa_debug save/vtextgqa_debug_abinet/vtextgqa_t2s_13/best.ckpt val

bash scripts/val_t2s_abinet.sh 0,1 vtextgqa_debug save/vtextgqa_debug_abinet/vtextgqa_t2s_13/best.ckpt inference
```
Note: you can access the checkpoint: [T2S_abinet](https://drive.google.com/file/d/1ye-E9L_9HbHiPDLRCpRYpwLdpyjP339q/view?usp=drive_link) and [T2S_clipocr](https://drive.google.com/file/d/1YjFLcCemcD-KVBqTojVre3YZzHQCnFQK/view?usp=drive_link).

## Visualization (VText-GQA)
<p align="center">
  <img src="https://github.com/zhousheng97/T2S/blob/main/image/visualization.png" alt="Visualization">
</p>


## Acknowledgements
The model implementation of our T2S is inspired by [MMF](https://github.com/facebookresearch/mmf).  The dataset of our VText-GQA is inspired by [M4-ViteVQA](https://github.com/bytedance/VTVQA). 

## Citation
If you found this work useful, consider giving this repository a star and citing our papers as follows:

```

```
