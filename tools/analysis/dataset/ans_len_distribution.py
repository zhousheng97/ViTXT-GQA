import json


set_ = 'val'
qa_folder = '/data/zsheng/Data_T5_ViteVQA/data/m4vitevqa/ground_annotation/' + 'qa_sub_t1s2'+ set_ +'.json'

with open(qa_folder, 'r') as file:
  qa_file = json.load(file)

an_len_list = []

for i, element in enumerate(qa_file['data']):
  answer = element['answers']
  an_len = len(answer[0].split())
  an_len_list.append(an_len)


import statistics
average_ans_percentage = statistics.mean(an_len_list)
median_ans_percentage = statistics.median(an_len_list)
print(f"{set_} set, que num:{len(an_len_list)}")
print(f"{set_} set, ans word ave: {average_ans_percentage}, med:{median_ans_percentage}")