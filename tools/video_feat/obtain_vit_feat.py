import os
from PIL import Image
import glob
import numpy as np
import torch

from transformers import ViTImageProcessor, ViTModel
from transformers import T5Tokenizer, BertTokenizer


device = torch.device("cuda:0")  

processor = ViTImageProcessor.from_pretrained('../huggingface/vit-large-patch16-224-in21k')
model = ViTModel.from_pretrained('../huggingface/vit-large-patch16-224-in21k')
bert_tokenizer = BertTokenizer.from_pretrained('../huggingface/bert-base-uncased')

# Move processor and model to GPU
model = model.to(device)

source_path = 'root/data/fps10_frames'
end_path = 'root/data/fps10_video_vit_feat'


for video in sorted(os.listdir(source_path)):
  if 0<= int(video)<=1000:
  # if int(video) == 0:
    print(f"video:{video}")
    video_path = os.path.join(source_path, video)

    save_folder = os.path.join(end_path, video)
    if not os.path.exists(save_folder):
            os.makedirs(save_folder, exist_ok=True)
    elif len(os.listdir(save_folder)) == len(os.listdir(video_path)):   
          print('video feature exists.')
          continue
    
    for frames in sorted(os.listdir(video_path)):
      frame_path = os.path.join(video_path, frames)
      feat_path = os.path.join(end_path, video, frames)

      if os.path.exists(feat_path):
          print(f'{frames} feature exists.')
          continue

      image = Image.open(frame_path) #.convert("RGB")
      # ViT for frame feature
      inputs = processor(images=image, return_tensors="pt").to(device)
      with torch.no_grad():
        outputs = model(**inputs).last_hidden_state[:,0,:] # [197, 1024] -> [1, 1024]
        outputs = outputs.detach().cpu().numpy()   
    
      save_path = os.path.join(save_folder, frames.split('.')[0]+'.npy')
      np.save(save_path, outputs)
