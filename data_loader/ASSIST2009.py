import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader


if torch.cuda.is_available():
    from torch.cuda import FloatTensor
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
    device = "cuda"
else:
    from torch import FloatTensor
    device = "cpu"

Dataset_dir = 'Dataset/skill_builder_data_2009.csv'

class Preprocess2009(Dataset):
    def __init__(self, seq_len=None, data_dir=Dataset_dir):
        super().__init__()

        self.data_dir = data_dir
        self.q_seqs, self.r_seqs, self.num_s, self.num_q = self.preprocess()
        if seq_len:
            self.q_seqs, self.r_seqs = self.match_sequence(self.q_seqs, self.r_seqs, seq_len)
        self.len = len(self.q_seqs)

    def __getitem__(self, idx):
        return self.q_seqs[idx], self.r_seqs[idx]

    def __len__(self):
        return self.len

    def preprocess(self):
        df=pd.read_csv(self.data_dir, encoding = "ISO-8859-1", low_memory=False)
        df = df.dropna(subset=['skill_name']).drop_duplicates(['order_id', 'skill_name']).sort_values(by=['order_id'])

        student_list = np.unique(df['user_id'].values)
        question_list = np.unique(df['skill_name'].values) # len(question_list) --> 110 questions
        num_s = len(student_list) # 학생 수
        num_q = len(question_list) # 문제 수

        s2idx = {u: idx for idx, u in enumerate(student_list)}
        q2idx = {u: idx for idx, u in enumerate(question_list)}

        q_seqs = []
        r_seqs = []

        for student in student_list:
            df_student = df[df['user_id'] == student]

            q_seq = np.array([q2idx[q] for q in df_student['skill_name'].values]) # 문제 넘버를 0부터 시작하게함
            r_seq = df_student['correct'].values

            q_seqs.append(q_seq)
            r_seqs.append(r_seq)
        
        return q_seqs, r_seqs, num_s, num_q
    
    def match_sequence(self, q_seqs, r_seqs, seq_len, pad_val=-1):
        match_q_seqs = []
        match_r_seqs = []
        for q_seq, r_seq in zip(q_seqs, r_seqs):
            i = 0
            while i + seq_len + 1 < len(q_seq):
                match_q_seqs.append(q_seq[i:i + seq_len + 1])
                match_r_seqs.append(r_seq[i:i + seq_len + 1])
                i += seq_len + 1

            match_q_seqs.append(
                np.concatenate(
                    [q_seq[i:],
                    np.array( [pad_val] * ( i + seq_len + 1 - len(q_seq) ) )]
                )
            )
            match_r_seqs.append(
                np.concatenate(
                    [r_seq[i:],
                    np.array([pad_val] * ( i + seq_len + 1 - len(q_seq) ))]
                )
            )
        

        return match_q_seqs, match_r_seqs

def collate_fn(batch, pad_val=-1):
    problem_seqs = []
    answer_seqs = []
    problem_shift_seqs = []
    answer_shift_seqs = []

    for q_seq, r_seq in batch:
        problem_seqs.append(FloatTensor(q_seq[:-1]))
        answer_seqs.append(FloatTensor(r_seq[:-1]))
        problem_shift_seqs.append(FloatTensor(q_seq[1:]))
        answer_shift_seqs.append(FloatTensor(r_seq[1:]))

    problem_seqs = pad_sequence(
        problem_seqs, batch_first=True, padding_value=pad_val
    ) # [batch, 배치 내 가장 긴 시퀀스의 길이]
    answer_seqs = pad_sequence(
        answer_seqs, batch_first=True, padding_value=pad_val
    ) # [batch, 배치 내 가장 긴 시퀀스의 길이]
    problem_shift_seqs = pad_sequence(
        problem_shift_seqs, batch_first=True, padding_value=pad_val
    ) # [batch, 배치 내 가장 긴 시퀀스의 길이]
    answer_shift_seqs = pad_sequence(
        answer_shift_seqs, batch_first=True, padding_value=pad_val
    ) # [batch, 배치 내 가장 긴 시퀀스의 길이]

    mask_seqs = (problem_seqs != pad_val) * (problem_shift_seqs != pad_val) # 어디가 패딩된건지 골라낼 마스크

    problem_seqs =problem_seqs * mask_seqs 
    answer_seqs = answer_seqs * mask_seqs
    problem_shift_seqs = problem_shift_seqs * mask_seqs
    answer_shift_seqs = answer_shift_seqs * mask_seqs

    return problem_seqs, answer_seqs, problem_shift_seqs, answer_shift_seqs, mask_seqs

def assist2009(train_ratio, batch_size, seq_len):
    data = Preprocess2009(seq_len)
    train_size = int(len(data) * train_ratio)
    val_size = len(data) - train_size

    train_dataset, test_dataset = torch.utils.data.random_split(data, [train_size, val_size], generator=torch.Generator(device=device))
    train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            collate_fn=collate_fn, generator=torch.Generator(device=device)
        )
    val_loader = DataLoader(
        test_dataset, batch_size=val_size, shuffle=False,
        collate_fn=collate_fn, generator=torch.Generator(device=device)
        )
    return train_loader, val_loader, data.num_q
    