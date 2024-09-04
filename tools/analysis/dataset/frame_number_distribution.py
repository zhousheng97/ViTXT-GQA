import os
import json
import cv2

set_ ='test'
folder_path = '/data/zsheng/Data_T5_ViteVQA/data/fps10_video'

train_video_folder = '/data/zsheng/Data_T5_ViteVQA/data/fps10_ocr_detection/'+set_
video_id_list = os.listdir(train_video_folder)
print("video number = ", len(video_id_list))

avg_frames_num, avg_total_num = 0,0
frames_list = []
frame_files_list = os.listdir(folder_path)

excel_list = [['Video ID', 'Frame Number']]

for video_id in video_id_list:
    video_id = video_id.split('.')[0]

    video_path = os.path.join(folder_path, video_id+'.mp4')
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    frames_length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    video.release()
    frames_list.append(frames_length)
    excel_list.append([video_id, frames_length])


import statistics
average_frames_percentage = statistics.mean(frames_list)
median_frames_percentage = statistics.median(frames_list)
print(f"{set_} set, frame ave: {average_frames_percentage}, med:{median_frames_percentage}")

import openpyxl

def write_list_to_excel(data, filename):
    workbook = openpyxl.Workbook()
    sheet = workbook.active

    for row in data:
        sheet.append(row)

    workbook.save(filename)


write_list_to_excel(excel_list, 'Distribution_of_frame_number_on_'+ set_ +'.xlsx')