'''Calculate OCRs distribution'''

import os
import statistics
import numpy as np
import matplotlib.pyplot as plt

ocr_info_path = '/data/zsheng/Data_T5_ViteVQA/data/fps10_ocr_detection'
set_ = 'val'
videos_path = os.path.join(ocr_info_path, set_)
videos_folder = os.listdir(videos_path)

total_ocr_count = 0  # 
total_video_count = len(videos_folder)  

ocr_counts_total_list = []
ocr_counts_perframe_list = []

for video in videos_folder:
    video_path = os.path.join(videos_path, video)
    ocr_infos = np.load(video_path, allow_pickle=True).item()
  
    video_ocr_count = 0      
    video_frame_count = len(ocr_infos)  

    for idx,ocr in enumerate(ocr_infos):
        ocr_count = ocr_infos[ocr]
        video_ocr_count += len(ocr_count)
    
    if video_frame_count > 0:
        video_average_ocr_count = video_ocr_count / video_frame_count
    else:
        video_average_ocr_count = 0
    ocr_counts_perframe_list.append(video_average_ocr_count)
    ocr_counts_total_list.append(video_ocr_count)
    
    total_ocr_count += video_average_ocr_count

print()


frame_mean = sum(ocr_counts_perframe_list) / len(ocr_counts_perframe_list)
frame_median = statistics.median(ocr_counts_perframe_list)
print(f"In video frames, OCR token ave in {set_} set:", frame_mean)
print(f"In video frames, OCR token med in {set_} set:", frame_median)
print()

video_mean = sum(ocr_counts_total_list) / len(ocr_counts_total_list)
video_median = statistics.median(ocr_counts_total_list)
print(f"In video, OCR token med in {set_} set:", video_mean)
print(f"In video, OCR token med in {set_} set:", video_median)
print()


