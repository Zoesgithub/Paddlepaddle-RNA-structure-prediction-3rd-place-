import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
from torch_geometric.nn.conv import MessagePassing
import torch
import numpy as np

class supermodel(nn.Module):
    def get_output(self, seq, nedge, edgeattr, pedge):
        node=self.embedding(seq)

        emb=self.forward(node, edgeattr, nedge)
        emb=self.out_linear(emb)

        lemb=emb[pedge[0]]
        remb=emb[pedge[1]]
        pred_edge=self.sig(self.out_edge_linear(torch.cat([lemb, remb], 1)))
        pred_node=self.sig(self.out_node_linear(emb))
        return pred_edge, pred_node

    def train_onestep(self, data, use_label=True):
        self.train()
        self.zero_grad()

        seq=data.x.cuda()
        nbond=data.y.cuda()
        nattr=data.y_attr.cuda()
        pbond=data.z.cuda()
        pattr=data.z_attr.cuda()
        id=data.id
        label=data.l.cuda()
        nlabel=data.x_label.cuda()

        pred_edge, pred_node=self.get_output(seq, nbond, nattr.reshape(-1, 1), pbond)
        pred_edge=pred_edge.reshape(-1)
        pred_node=pred_node.reshape(-1)

        loss=-pattr.reshape(-1)*torch.log(pred_edge+1e-8)-(1-pattr.reshape(-1))*torch.log(1-pred_edge+1e-8)
        loss=loss.sum()*0.01
        nloss=-nlabel*torch.log(pred_node+1e-8)-(1-nlabel)*torch.log(1-pred_node+1e-8)
        loss+=nloss.sum()
        if use_label and len(label)>0:
            upp=1-pred_node
            lloss=((label-upp)**2).sum()
            loss+=lloss*10.0
        loss.backward()
        self.optimizer.step()
        return  loss

    def Eval(self, dataloader):
        self.eval()
        acc=0.0
        size=0.0
        rmsd=0.0
        ret=[]
        for data in dataloader:
            seq=data.x.cuda()
            nbond=data.y.cuda()
            nattr=data.y_attr.cuda()
            pbond=data.z.cuda()
            pattr=data.z_attr.cuda()
            id=data.id
            label=data.l.cuda()
            x_label=data.x_label.cuda()

            with torch.no_grad():
                pred_edge, pred_node=self.get_output(seq, nbond, nattr.reshape(-1,1), pbond)
                pred_edge=pred_edge.reshape(-1)
                pred_node=pred_node.reshape(-1)
            ret.append([id, pred_node.detach().cpu().numpy()])
            acc+=(((pred_node>0.5).float()==x_label).float()).sum()
            size+=len(x_label)
            if len(label)>0:
                rmsd+=((1-pred_node-label)**2).mean().sqrt()
            else:
                rmsd+=0
            #print(upp.max(), upp.min(), upp.shape, label.shape, pred_edge.shape, pred_edge.max(), pred_edge.min(), pattr.shape)

        print(pred_edge.max(), pred_edge.min())
        acc=acc/size
        rmsd=rmsd/len(dataloader)
        return acc, rmsd, ret
