import os

import numpy as np
import torch

from torch.nn import Module, Embedding, Linear, Dropout, Parameter
from sklearn import metrics
from tqdm import tqdm
import wandb

class DKVMN(Module):
    def __init__(self, num_q, s_dim, m_size):
        '''
            num_q: 총 문항 수
            s_dim: state vector의 dim 사이즈. 이 구현에선 dk,dv를 동일하게 놓겠음
            m_size: memory size(N개의 개념)
        '''
        super().__init__()
        self.num_q = num_q
        self.s_dim = s_dim
        self.m_size = m_size

        self.k_embedding = Embedding(self.num_q, self.s_dim)
        self.Mk = Parameter(torch.Tensor(self.m_size, self.s_dim))
        self.Mv0 = Parameter(torch.Tensor(self.m_size, self.s_dim))

        torch.nn.init.xavier_normal_(self.Mk)
        torch.nn.init.xavier_normal_(self.Mv0)

        self.f_layer = Linear(self.s_dim * 2, self.s_dim)
        self.p_layer = Linear(self.s_dim, 1)

        self.v_embedding = Embedding(self.num_q * 2, self.s_dim)

        self.e_layer = Linear(self.s_dim, self.s_dim)
        self.a_layer = Linear(self.s_dim, self.s_dim)       

    def forward(self, question, respond):
        # correlation weight
        k = self.k_embedding(question) #[batch, seq_len, s_dim]
        w = torch.softmax(torch.matmul(k, self.Mk.T), dim=-1) # [batch, seq_len, m_size]

        # write process
        x = question + self.num_q * respond
        v = self.v_embedding(x) # [batch, seq_len, s_dim]
        e = torch.sigmoid(self.e_layer(v)) # [batch, seq_len, s_dim]
        a = torch.tanh(self.a_layer(v))# [batch, seq_len, s_dim]

        batch_size = question.shape[0]
        Mvt = self.Mv0.unsqueeze(0).repeat(batch_size, 1, 1) # weight sum하기 위해 배치사이즈만큼 추가생성

        
        ## 각 시퀀스에서 시점 t별로 진행되기 때문에 t별 Mv를 저장하기위함
        Mv = [Mvt]
        for wt, et, at in zip(w.permute(1, 0, 2), e.permute(1, 0, 2), a.permute(1, 0, 2)):
            '''
                seq_len 만큼 반복하면서 Mv 생성
                wt : [batch, m_size] -->[batch, m_size, s_dim]
                et : [batch, s_dim] -->[batch, m_size, s_dim]
                at : [batch, s_dim] -->[batch, m_size, s_dim]
            '''
            Mvt_tilde = Mvt * (1 - wt.unsqueeze(2) * et.unsqueeze(1)) # broadcast을 이용해 weight sum을 할 수 있음 
            Mvt = Mvt_tilde + (wt.unsqueeze(2) * at.unsqueeze(1))
            Mv.append(Mvt)
        Mv = torch.stack(Mv, dim=1) #Mv : [batch, seq_len+1, m_size, s_dim]

        # read process
        r = (w.unsqueeze(-1) * Mv[:, :-1]).sum(2) # [batch, seq_len, s_dim]
        f = torch.tanh(
            self.f_layer(torch.cat([r,k], dim=-1))  # [batch, seq_len, s_dim]
        )
        p = torch.sigmoid(
            self.p_layer(f)  # [batch, seq_len, 1]
        )
        pred = p.squeeze(-1) # [batch, seq_len]
        return pred
    
    def run_train(self, train_loader, val_loader, epochs, optimizer):
        wandb.init(project='DKVMN', entity="lemma17")
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
                    output = self(problem_seqs.long(), answer_seqs.long()) # [batch, seq_len]
                    
                    output = torch.masked_select(output, mask_seqs).detach().cpu() # [batch_size, mask_seq_length]
                    target = torch.masked_select(answer_seqs, mask_seqs).float().detach().cpu() # [batch_size, mask_seq_length]

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

                output = self(problem_seqs.long(), answer_seqs.long())
                
                output = torch.masked_select(output, mask_seqs)
                target = torch.masked_select(answer_seqs, mask_seqs).float()

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
