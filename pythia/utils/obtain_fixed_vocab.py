import yaml

class FixedVocab:
    def __init__(self, args):
        dataset = args.datasets
        with open(args.config) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        # './data/m4c_vocabs/textvqa/fixed_answer_vocab_textvqa_5k.txt'
        fixed_ans_path = config['dataset_attributes'][dataset]['processors']['answer_processor']['params']['vocab_file']
        fixed_ans_path = '/data/zsheng/Data_T5_ViteVQA/data/{}/{}'.format(dataset, fixed_ans_path)
        with open('/data/zsheng/Data_T5_ViteVQA/data/fixed_answer_vocab.txt', 'w+', encoding='utf-8') as f:
            f.writelines(fixed_ans_path)
       