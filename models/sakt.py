import os

import numpy as np
import torch

from torch.nn import Module, Embedding, Linear, Dropout, MultiheadAttention, LayerNorm
from sklearn import metrics
from tqdm import tqdm
import wandb

class SAKT(Module):
    def __init__(self, num_q, d_dim, len_n, head_dim, dropout_ratio):
        super().__init__()
        self.num_q = num_q
        self.d_dim = d_dim # d_dim % head_dim = 0 이어야함
        self.len_n = len_n
        self.head_dim = head_dim
        self.dropout = dropout_ratio

        self.M = Embedding(self.num_q * 2, self.d_dim)
        self.E= Embedding(self.num_q, self.d_dim)
        self.P = Embedding(self.num_q, self.d_dim)

        self.self_attention = MultiheadAttention(self.d_dim, self.head_dim, dropout=self.dropout)
        self.Dropout = Dropout(self.dropout)
        self.atten_layernorm = LayerNorm(self.d_dim)

        self.ffn = torch.nn.Sequential(
            Linear(self.d_dim, self.d_dim),
            torch.nn.ReLU(),
            Dropout(self.dropout),
            Linear(self.d_dim, self.d_dim),
            Dropout(self.dropout),
        )
        self.f_layernorm = LayerNorm(self.d_dim)

        self.pred = Linear(self.d_dim, 1)

    def forward(self, question, respond, question_shift):
        device = question.device
        x = question + respond * self.num_q
        src_len = question.shape[1]
        batch_size = question.shape[0]
        
        # Embedding layer
        M = self.M(x).permute(1, 0, 2) # [batch, src_len, hidden_dim] -> [src_len, batch, hidden_dim]
        E = self.E(question_shift).permute(1, 0, 2) # [batch, src_len, hidden_dim] -> [src_len, batch, hidden_dim]
        pos = torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1)
        P = self.P(pos).permute(1, 0, 2) # [batch, src_len, hidden_dim] -> [src_len, batch, hidden_dim]

        # positional encodeing
        M_hat = M + P

        # multi-head-attention
        mask = torch.triu(torch.ones(E.shape[0], M.shape[0]), diagonal=1).bool() # torch.tril이 아닌 이유: MHA내부에서 true 값에 -inf를 masked_fill_하기 때문에!
        S, atten_weight = self.self_attention(E, M_hat, M_hat, attn_mask=mask)

        # add + norm
        S = self.Dropout(S)
        S = S.permute(1, 0, 2) # [src_len, batch, hidden_dim] -> [batch, src_len, hidden_dim]
        M = M.permute(1, 0, 2) # [src_len, batch, hidden_dim] -> [batch, src_len, hidden_dim]
        E = E.permute(1, 0, 2) # [src_len, batch, hidden_dim] -> [batch, src_len, hidden_dim]
        S = self.atten_layernorm(S + M + E) # [batch, src_len, hidden_dim]

        # FFN
        F = self.ffn(S)
        F = self.f_layernorm(F + S) # [batch, src_len, hidden_dim]

        # pred layer
        pred = torch.sigmoid(self.pred(F)).squeeze() # [batch, src_len]

        return pred, atten_weight
    
    def run_train(self, train_loader, val_loader, epochs, optimizer):
        wandb.init(project='SAKT', entity="lemma17")
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
                    output, _ = self(problem_seqs.long(), answer_seqs.long(), problem_shift_seqs.long()) # [batch, seq_len]
                    
                    output = torch.masked_select(output, mask_seqs).detach().cpu() # [batch_size, mask_seq_length]
                    target = torch.masked_select(answer_shift_seqs, mask_seqs).float().detach().cpu() # [batch_size, mask_seq_length]

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

                output, _ = self(problem_seqs.long(), answer_seqs.long(), problem_shift_seqs.long())
                
                output = torch.masked_select(output, mask_seqs)
                target = torch.masked_select(answer_shift_seqs, mask_seqs).float()

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