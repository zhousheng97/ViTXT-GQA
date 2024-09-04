import torch
import editdistance  # install with `pip install editdistance`

class ANLSEvaluator:
    def __init__(self, fixed_ans_path, theta=0):
        self.get_edit_distance = editdistance.eval
        self.theta = theta
        with open(fixed_ans_path) as f:
            fixed_ans = f.readlines()
        self.fixed_ans = [i.replace('\n', '') for i in fixed_ans]
        self.fixed_ans_size = len(self.fixed_ans)

    def idx2word(self, idx, ocr_tokens):
        if idx >= self.fixed_ans_size:
            return ocr_tokens[idx-self.fixed_ans_size]
        else:
            return self.fixed_ans[idx] 

    def get_anls(self, s1, s2):
        s1 = s1.lower().strip()
        s2 = s2.lower().strip()
        iou = 1 - self.get_edit_distance(s1, s2) / max(len(s1), len(s2))
        anls = iou if iou >= self.theta else 0.
        return anls

    def compute_anls(self, predicted_scores, gt_answers, ocr_tokens):
        batch, T = predicted_scores.size()
        batch_anls = []
        for i in range(batch):
            i_ans = []
            for j in range(T):
                pred_idx = predicted_scores[i, j]
                if pred_idx>2:
                    i_ans.append(self.idx2word(pred_idx, ocr_tokens[i]))
            i_ans = ' '.join(i_ans)
            anls = max(self.get_anls(i_ans, gt) for gt in set(gt_answers[i]))
            batch_anls.append(anls)
        return batch_anls

if __name__ == '__main__':
    model = STVQAANLSEvaluator()
    a = 'sdfs sdf' 
    b = 'sdfs sd1f' 
    print(model.get_anls(a,b))
