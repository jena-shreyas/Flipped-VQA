import torch
from .base_dataset import BaseDataset
import json
import pandas as pd
import random

class CausalVidQA(BaseDataset):
    def __init__(self, args=None, tokenizer=None, split='train'):
        super().__init__(args, tokenizer, split)
        self.data = pd.read_csv(f'./data/causalvidqa/{split}.csv')
        ##########
        self.data = self.data.iloc[:1500]
        ##########
        # self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        # self.features = torch.load('./data/causalvidqa/clipvitl14_causalvidqa.pth', map_location = self.device)
        ##########

        # The issue with  - all the video features were being dumped on GPU !
        # Since the model will be loaded to the GPU, also loading the whole features there leaves little space for model loading !
        # Also, since the whole data is loaded on GPU already, num_workers > 0 initializes CUDA separately and creates
        #      new subprocesses on GPU to create batches and send to GPU. But the whole data is already in GPU !!
        # So, CUDA gets initialized twice - once while loading whole data (self.features) to GPU, the other when initializing workers for dataloader.

        # This multiple CUDA initialization causes CUDA initialization errors !!!

        # A better way out - load the data on CPU RAM, and only send batches to GPU at a time.
        # In order to send sufficient batches for GPU to get saturated, increase num_workers till the performance drops.
        # ALso, set pin_memory = True for this to work !
        self.features = torch.load('./data/causalvidqa/clipvitl14_causalvidqa.pth')
        self.features = {k:v.to('cpu') for k, v in self.features.items()}  # since all tensors in features dict were in CUDA by default, bring them to CPU first acc to the above argument
        self.answer_mapping = {0: '(A)',1: '(B)',2: '(C)',3: '(D)',4: '(E)'}
        self.num_options = 5
        self.qtype_mapping = {'descriptive': 1, 'explanatory': 2, 'predictive': 3, 'counterfactual': 4}
        print(f"Num {split} data: {len(self.data)}")
        
    def __len__(self):
        return len(self.data)

    def _get_text(self, idx):
        row = self.data.iloc[idx]
        question = row['question'].strip()
        if question[-1] != "?":
            question = str(question) + "?"

        options = [row[f'a{i}'] for i in range(self.num_options)]

        q_text = f"Question: {question}\n"
        o_text = "Choices: \n"
        for i in range(self.num_options):
            o_text += f"{self.answer_mapping[i]} {options[i]}\n"
        
        a_text = "Answer: The answer is "
        text = {'q_text': q_text, 'o_text': o_text, 'a_text': a_text, 'options': options}
        return text

    def _get_video(self, video_id):
        if video_id not in self.features:
            print(video_id)
            video = torch.zeros(1, self.features_dim)
        else:
            video = self.features[video_id].float()
        if len(video) > self.max_feats:
            sampled = []
            for j in range(self.max_feats):
                sampled.append(video[(j * len(video)) // self.max_feats])
            video = torch.stack(sampled)
            video_len = self.max_feats
        elif len(video) < self.max_feats:
            video_len = len(video)
            video = torch.cat([video, torch.zeros(self.max_feats - video_len, self.features_dim)], dim=0)
        else:
            video_len = self.max_feats

        return video, video_len

    def __getitem__(self, idx):
        try:
            # # vid = self.data['video'].values[idx]
            # print("data len: ", len(self.data))
            row = self.data.iloc[idx]
            question = row['question']
            qtype = self.qtype_mapping[row['qn_type']]
            vid = row['video_id']
            answer = int(row['answer'])
            if answer == -1:
                answer = random.randint(0, 4)
            text = self._get_text(idx)
            # print("Text : ")
            # print(text)
            # print("Answer :")
            # print(answer)
            text_id, label, video_start, video_index, label_mask = self._get_text_token(text, answer)
            video, video_len = self._get_video(f'{vid}')
            return {"vid": vid, "video": video, "video_len": video_len, "text": text, "text_id": text_id, "label": label, "video_start": video_start,
                    "video_index": video_index, "label_mask": label_mask, "qid": idx, "answer": answer, "qtype": qtype}
        except:
            return None

