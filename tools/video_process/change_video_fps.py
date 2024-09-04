'''
  Change the number of frames per second of video from fps=30 to fps=10, and use video at fps=10 for data annotation
'''

import cv2
import os
import json
import moviepy
from moviepy.editor import VideoFileClip, ImageSequenceClip


set_ = 't1s2test'

frame_folder = 'root/data/raw_frames'
json_folder = 'root/data/m4vitevqa/qa_annotation/'+'ViteVQA_0.0.2_'+set_+'.json'


output_folder = 'root/data/fps10_video/'
if not os.path.exists(output_folder):    
    os.makedirs(output_folder, exist_ok=True)

json_file = json.load(open(json_folder,'r'))

for _, value in enumerate(json_file['data']):
    
    video_id =  value['video_id']
    video_path = os.path.join(output_folder,video_id+'.mp4')
    if os.path.exists(video_path):
        print(f"Video path '{video_path}' exist.")
        continue
    else:
        frame_path = os.path.join(frame_folder, video_id)
        frame_files = sorted(os.listdir(frame_path), key=lambda x: int(os.path.splitext(x)[0]))
        frames = [frame_path+ '/' + frame_files for frame_files in frame_files]
        
        new_clip = ImageSequenceClip(frames, fps=30)
        new_clip.write_videofile(os.path.join(output_folder, f"{video_id}.mp4"), codec='libx264', fps=10) 
