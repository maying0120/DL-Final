import sys
import os
import pickle

import torch
from torch.utils.data import Dataset

from transformers import BertTokenizer


class MyDataset(Dataset):

    def __init__(self, data):
        super(MyDataset, self).__init__()
        self.data = data
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        prev_v, v, next_v = item['prev_v'], item['v'], item['next_v']
        prev_t, t, next_t = item['prev_t'], item['t'], item['next_t']
        label = item['label']
        start = item['start']
        end = item['end']
        duration = item['duration']
        video_id = item['video_id']
        return prev_v, v, next_v, prev_t, t, next_t, label, start, end, duration, video_id

    def collate_fn(self, items):

        to_return = {
            'prev_v': [],
            'v': [],
            'next_v': [],
            'prev_t': [],
            't': [],
            'next_t': [],
            'label': [],
            'start': [],
            'end': [],
            'duration': [],
            'video_id': []
        }

        for prev_v, v, next_v, prev_t, t, next_t, label, start, end, duration, video_id in items:
            to_return['prev_v'].append(prev_v)
            to_return['v'].append(v)
            to_return['next_v'].append(next_v)
            to_return['prev_t'].append(prev_t)
            to_return['t'].append(t)
            to_return['next_t'].append(next_t)
            to_return['label'].append(label)
            to_return['start'].append(start)
            to_return['end'].append(end)
            to_return['duration'].append(duration)
            to_return['video_id'].append(video_id)

        to_return['prev_v'] = torch.nn.utils.rnn.pad_sequence(to_return['prev_v'], batch_first=True).cuda()
        to_return['prev_v_mask'] = mask_data(to_return['prev_v'][:, :, 0]).cuda()
        to_return['v'] = torch.nn.utils.rnn.pad_sequence(to_return['v'], batch_first=True).cuda()
        to_return['v_mask'] = mask_data(to_return['v'][:, :, 0]).cuda()
        to_return['next_v'] = torch.nn.utils.rnn.pad_sequence(to_return['next_v'], batch_first=True).cuda()
        to_return['next_v_mask'] = mask_data(to_return['next_v'][:, :, 0]).cuda()
        prev_t_tok = self.tokenizer(to_return['prev_t'], padding=True, return_tensors='pt')
        to_return['prev_t'] = prev_t_tok.input_ids.cuda()
        to_return['prev_t_mask'] = prev_t_tok.attention_mask.cuda()
        t_tok = self.tokenizer(to_return['t'], padding=True, return_tensors='pt')
        to_return['t'] = t_tok.input_ids.cuda()
        to_return['t_mask'] = t_tok.attention_mask.cuda()
        next_t_tok = self.tokenizer(to_return['next_t'], padding=True, return_tensors='pt')
        to_return['next_t'] = next_t_tok.input_ids.cuda()
        to_return['next_t_mask'] = next_t_tok.attention_mask.cuda()
        label_tok = self.tokenizer(to_return['label'], padding=True, return_tensors='pt')
        to_return['label'] = label_tok.input_ids.cuda()
        to_return['label_mask'] = label_tok.attention_mask.cuda()

        return to_return
