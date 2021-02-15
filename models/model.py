import math, random

import numpy as np
import torch
import torch.nn as nn
from .egsage import egsage


# from paddle.fluid.dygraph.learning_rate_scheduler import LearningRateDecay

class Model(nn.Module):
    def __init__(self, lr
                 ):
        super(Model, self).__init__()
        self.seq_embedding = nn.Embedding(4, 128)
        self.bra_embedding=nn.Embedding(3, 128)
        self.net=nn.LSTM(128, 128*4, 8, bidirectional=True, dropout=0.15)
        self.inp_fc=nn.Linear(128, 128)
        self.seq_fc=nn.Linear(128, 128)
        self.fc=nn.Linear(256, 128)

        self.out_linear=nn.Sequential(nn.Linear(256*4, 128), nn.ReLU(), nn.Linear(128, 2))
        self.gcn1=egsage(128, 128, 1, nn.ReLU,1, True, "mean")
        self.gcn2 = egsage(128, 128, 1, nn.ReLU, 1, True, "mean")

        self.relu=nn.ReLU()
        self.sig=nn.Sigmoid()
        self.soft=nn.Softmax(-1)
        self.optimizer=torch.optim.Adam(self.parameters(), lr=lr)

    def forward(self, seq, dot, slabel):
        emb_seq=self.seq_embedding(seq)
        '''edge=[]
        edge_attr=[]
        stack=[]
        assert len(seq)==1

        for i,x in enumerate(dot[0][0]):
            if i>0:
                edge.append([i, i-1])
                edge.append([i-1, i])
                edge_attr.append(1.0)
                edge_attr.append(1.0)
            if x==0:
                stack.append(i)
            elif x==1:
                l=stack.pop(-1)
                edge.append([l, i])
                edge.append([i, l])
                edge_attr.append(2.0)
                edge_attr.append(2.0)
        edge=torch.tensor(edge).cuda().permute(1, 0)
        edge_attr=torch.tensor(edge_attr).cuda().reshape(-1,1)'''

        emb_dot=(self.bra_embedding(dot)).squeeze(1)#reshape(-1, seq.shape[1], 32)

        emb_dot=nn.ReLU()(self.inp_fc(emb_dot))
        emb_seq=nn.ReLU()(self.seq_fc(emb_seq))

        emb=self.relu(self.fc(torch.cat([emb_seq, emb_dot], 2))).permute(1, 0, 2)

        #emb=self.gcn1(emb.squeeze(1), edge_attr, edge).unsqueeze(1)
        #emb = self.gcn2(emb.squeeze(1), edge_attr, edge).unsqueeze(1)

        emb, _=self.net(emb)
        out=self.out_linear(emb).permute(1, 0, 2)
        out=self.soft(out)

        return out[:, :, 0]

    def train_onestep(self, data, test=False):
        self.train()
        self.optimizer.zero_grad()
        id, x, y, Size, stru, slabel = data


        x = x.cuda()
        y = y.cuda()

        out= self.forward(x, stru.cuda(), slabel)

        loss = ((y - out) ** 2 * (y > -0.1).float()).mean(1).sqrt().add(
            1e-9).sum()

        loss.backward()
        self.optimizer.step()
        return loss

    def Eval(self, dataloader):
        self.eval()
        acc = 0.0
        size = 0.0
        rmsd = 0.0
        ret = []
        gt = []
        RSize = []
        for data in dataloader:
            id, x, y, Size, stru, slabel= data
            Size = Size.reshape(-1)
            assert len(Size) == len(x)

            x = x.cuda()
            with torch.no_grad():
                out = self.forward(x, stru.cuda(), slabel.cuda())

            for i, d in enumerate(id):
                ret.append([d, out[i].detach().cpu().numpy()])
                gt.append([d, y[i].detach().cpu().numpy()])
                RSize.append([d, Size[i].numpy().tolist()])
            rmsd += ((out - y.cuda()) ** 2).mean(1).sqrt().sum()
            size += len(x)

        acc = 0  # acc/size
        rmsd = rmsd / size
        return acc, rmsd, ret, gt, RSize

