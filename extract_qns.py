import os
import sys
import pandas as pd
import torch
import json
from tqdm import tqdm

def save_qa(save_path: str, df: pd.DataFrame, ratios: list):
    tr, val, _ = ratios
    len_train = int(tr * len(df))
    len_val = int(val * len(df))

    train_df = df.iloc[:len_train]
    val_df = df.iloc[len_train: len_train+len_val]
    test_df = df.iloc[len_train+len_val:]

    train_df.to_csv(os.path.join(save_path, 'train.csv'))
    val_df.to_csv(os.path.join(save_path, 'val.csv'))
    test_df.to_csv(os.path.join(save_path, 'test.csv'))
    

def extract_qns(data_path: str, save_path: str, video_ids: list, ratio: str):

    tr, val, test = [int(frac) for frac in ratio.split(':')]
    sum = tr + val + test
    tr = tr/sum
    val = val/sum
    test = test/sum

    questions = []
    qn_types = []
    vids = []
    a0 = []
    a1 = []
    a2 = []
    a3 = []
    a4 = []
    answers = []
    reasons = []
    num_qns = 0
    for video_id in tqdm(video_ids):
        try:
            with open(os.path.join(data_path, video_id, 'text.json'), 'r') as f:
                qns_dict = json.load(f)

            with open(os.path.join(data_path, video_id, 'answer.json'), 'r') as f:
                ans_dict = json.load(f)

            for qn_type in qns_dict:
                questions.append(qns_dict[qn_type]['question'])
                qn_types.append(qn_type)
                vids.append(video_id)
                a0.append(qns_dict[qn_type]['answer'][0])
                a1.append(qns_dict[qn_type]['answer'][1])
                a2.append(qns_dict[qn_type]['answer'][2])
                a3.append(qns_dict[qn_type]['answer'][3])
                a4.append(qns_dict[qn_type]['answer'][4])
                answers.append(ans_dict[qn_type]['answer'])
                if 'reason' in ans_dict[qn_type].keys():
                    reasons.append(ans_dict[qn_type]['reason'])
                else:
                    reasons.append("")

                if num_qns % 200 == 0:
                    data_dict = {
                        'question': questions,
                        'qn_type': qn_types,
                        'video_id': vids,
                        'a0': a0,
                        'a1': a1,
                        'a2': a2,
                        'a3': a3,
                        'a4': a4,
                        'answer': answers,
                        'reason': reasons
                    }
                    df = pd.DataFrame(data_dict)
                    save_qa(save_path, df, [tr, val, test])

                num_qns += 1
        except:
            print(video_id)

    data_dict = {
                    'question': questions,
                    'qn_type': qn_types,
                    'video_id': vids,
                    'a0': a0,
                    'a1': a1,
                    'a2': a2,
                    'a3': a3,
                    'a4': a4,
                    'answer': answers,
                    'reason': reasons
                }
    
    df = pd.DataFrame(data_dict)
    save_qa(save_path, df, [tr, val, test])


def main(args):
    data_path = args[1]
    save_path = args[2]
    ratio = "7:1:2"

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(os.getcwd())
    features = torch.load('../data/causalvidqa/clipvitl14_causalvidqa.pth', map_location=torch.device(device))
    video_ids = list(features.keys())
    extract_qns(data_path, save_path, video_ids, ratio)

if __name__ == '__main__':
    main(sys.argv)
