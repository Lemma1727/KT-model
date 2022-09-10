import os

import numpy as np
import torch

from torch.nn import Module, Embedding, LSTM, Linear, Dropout
from torch.nn.functional import one_hot, binary_cross_entropy
from sklearn import metrics
from tqdm import tqdm
import wandb

class DKT(Module):
    def __init__(self, num_problems, emb_size, hidden_size):
        super().__init__()
        self.num_problems = num_problems
        self.emb_size = emb_size
        self.hidden_size = hidden_size

        self.embedding = Embedding(self.num_problems*2, self.emb_size, padding_idx=0) # 각 문제당 정답 오답까지 표현하기 위해 두 배를 함
        self.lstm = LSTM(self.emb_size, self.hidden_size, batch_first=True)
        self.fc = Linear(self.hidden_size, self.num_problems) # pad value 0 때문
        self.dropout = Dropout()

    def forward(self, problem_seqs, answer_seqs):
        x = problem_seqs + self.num_problems * answer_seqs # 각 문제당 정답 오답까지 표현하기 위함 
        x = self.embedding(x)
        hidden, _ = self.lstm(x)
        out = self.fc(hidden)
        out = self.dropout(out)
        out = torch.sigmoid(out) # [batch_size, max_seq_length, num_problems]

        return out
    
    def run_train(self, train_loader, val_loader, epochs, optimizer):
        wandb.init(project='DKT', entity="lemma17")
        train_step=0
        loss_means = []
        aucs = []
        for i in range(1, epochs +1):
            # validation step
            with torch.no_grad():
                val_loss = 0.0
                val_auc = 0.0
                self.eval()
                
                for val_batch_idx, val_batch in enumerate(tqdm(val_loader, desc="validation")):
                    problem_seqs, answer_seqs, problem_shift_seqs, answer_shift_seqs, mask_seqs = val_batch

                    # forward
                    output = self(problem_seqs.long(), answer_seqs.long()) # [batch_size, max_seq_length, num_problems]
                    output = (output * one_hot(problem_shift_seqs.long(), self.num_problems)).sum(-1) # [batch_size, max_seq_length]

                    output = torch.masked_select(output, mask_seqs).detach().cpu() # [batch_size, mask_seq_length]
                    target = torch.masked_select(answer_shift_seqs, mask_seqs).detach().cpu() # [batch_size, mask_seq_length]

                    # loss & auc
                    val_loss += torch.nn.BCELoss()(output, target)

                    val_auc = metrics.roc_auc_score(y_true=target.numpy(), y_score=output.numpy())

            # valid step logging
            val_epoch_loss = val_loss / len(val_loader)
            val_epoch_auc = val_auc / len(val_loader)
            
            print("Epoch: {}, val loss: {}".format(i, val_epoch_loss))
            print("Epoch: {}, val auc: {}".format(i, val_epoch_auc))
            wandb.log({'Epoch': i, "val_loss": val_epoch_loss, "val_auc": val_epoch_auc})

            # train step
            loss_mean = []
            current_loss = 0.0
            for batch_idx, batch in enumerate(tqdm(train_loader, desc="training")):
                self.train()

                problem_seqs, answer_seqs, problem_shift_seqs, answer_shift_seqs, mask_seqs = batch
                problem_seqs, answer_seqs = problem_seqs, answer_seqs

                output = self(problem_seqs.long(), answer_seqs.long())
                output = (output * one_hot(problem_shift_seqs.long(), self.num_problems)).sum(-1)


                output = torch.masked_select(output, mask_seqs)
                target = torch.masked_select(answer_shift_seqs, mask_seqs)


                loss = torch.nn.BCELoss()(output, target)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                current_loss += loss.detach().cpu().numpy()

                if train_step % 100 == 0:
                    train_loss = current_loss / 100

                    print("{} >> trian_loss: {}".format(train_step, train_loss))
                    wandb.log({'train_step': train_step, "train_loss": train_loss})
                    loss_mean.append(current_loss)
                    current_loss = 0.0

                train_step += 1
                
            aucs.append(val_epoch_auc)
            loss_means.append(np.mean(loss_mean))
        return aucs, loss_means