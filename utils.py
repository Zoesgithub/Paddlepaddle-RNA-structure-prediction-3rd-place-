import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Data
import networkx as nx
import random
from torch_geometric.data import Dataset
import h5py
import os


Dict={"A":0, "C":1, "G":2, "U":3}
Labeldict={"(":0, ")":1, ".":2}
def aug(seq):
    stack=[]
    ret=seq.detach()
    for i, s in enumerate(seq):
        if s==0:
            stack.append(i)
        elif s==1:
            l=stack.pop(-1)
            r=i
            k=random.random()
            if k<0.1:
                ret[l]=2
                ret[r]=2
        else:
            assert s==2
    return ret

def parse_file(path, has_label=True):
    with open(path, "r") as f:
        content=f.readlines()
    n=len(content)
    r=0
    ret=[]
    while r<n:
        assert "id" in content[r]
        ID=content[r].replace("\n", "").split(" ")[0].split("_")[-1]

        r+=1
        SEQ=content[r].replace("\n", "").upper()
        length=len(SEQ)

        r+=1
        STRU=content[r].replace("\n", "")
        STRU=STRU.split(" ")
        if len(STRU)>1:
            ENE=float(STRU[-1].replace("(", "").replace(")", ""))
        else:
            ENE=-9999.0
        STRU=STRU[0]
        assert len(STRU)==length

        LABEL=[]
        if has_label and len(content[r+1])>2:
            while length>0:
                length-=1
                r+=1
                LABEL.append(float(content[r].replace("\n", "").split(" ")[-1]))
            assert len(LABEL)==len(SEQ)
        else:
            LABEL=[x=="." for x in STRU]
            #LABEL = []#[0 for x in STRU]
        r+=2
        SEQ=[Dict[x] for x in SEQ]
        STRU=[Labeldict[x] for x in STRU]
        ret.append([ID, torch.tensor(SEQ), torch.tensor(STRU), torch.tensor(LABEL), torch.tensor(ENE)])

    return ret

def pad_tensor(vec, pad, dim):
    """
    args:
        vec - tensor to pad
        pad - the size to pad to
        dim - dimension to pad

    return:
        a new tensor padded to 'pad' in dimension 'dim'
    """
    pad_size = list(vec.shape)
    pad_size[dim] = pad - vec.size(dim)
    return torch.cat([vec, torch.zeros(*pad_size)], dim=dim)


class PadCollate:
    """
    a variant of callate_fn that pads according to the longest sequence in
    a batch of sequences
    """

    def __init__(self, dim=0):
        """
        args:
            dim - the dimension to be padded (dimension of time in sequences)
        """
        self.dim = dim

    def pad_collate(self, batch):
        """
        args:
            batch - list of (tensor, label)

        reutrn:
            xs - a tensor of all examples in 'batch' after padding
            ys - a LongTensor of all labels in batch
        """
        # find longest sequence
        max_len = max(map(lambda x: x[1].shape[self.dim], batch))
        # pad according to max_len
        batch = list(map(lambda x:(x[0], pad_tensor(x[1], pad=max_len, dim=self.dim), pad_tensor(x[2], pad=max_len, dim=self.dim), x[3]), batch))
        # stack all
        id=[x[0]  for x in batch]
        xs = torch.stack([x[1] for x in batch], dim=0)
        ys = torch.stack([x[2] for x in batch], dim=0)
        ene=torch.stack([x[3] for x in batch], dim=0)
        return id, xs, ys, ene

    def __call__(self, batch):
        return self.pad_collate(batch)

class TrainDataSet(Dataset):
    def __init__(self, path, cut=False, aug=False):
        super(TrainDataSet, self).__init__()
        self.path=path
        self.data=parse_file(self.path)
        self.cut=cut
        self.padding_size=max([len(x[1]) for x in self.data])
        self.aug=aug

    def generate_train_data(self, idx):
        data=self.data[idx]
        id, seq, stru, label, ene=data
        seq=nn.functional.one_hot(seq.long(), 4).float()
        if self.aug:
            stru=aug(stru)
        stru=nn.functional.one_hot(stru.long(), 3).float()
        x=torch.cat([seq, stru], 1)
        if len(x)<self.padding_size:
            pad=torch.zeros([self.padding_size-len(x), 7])
            x=torch.cat([x, pad], 0)
            label=torch.cat([label, pad[:, 0]], 0)
        return [id, x, label, ene]

    def __getitem__(self, idx):
        return self.generate_train_data(idx)

    def __len__(self):
        return len(self.data)


def write_res(res, path):
    for x in res:
        p=os.path.join(path, "{}.predict.txt".format(x[0]))
        with open(p, "w") as f:
            for l in x[1]:
                f.writelines("{}\n".format(l))


if __name__=="__main__":
    temp=TrainDataSet("dev.txt")
    temp.generate_train_data(0)

