import numpy as np
import networkx as nx
import random
import h5py
import os
import RNA

Dict={"A":0, "C":1, "G":2, "U":3}
Labeldict={"(":0, ")":1, ".":2}
def store_structure(s, data):
    if s:
        data.append(s)


def parse_file(path, has_label=True):
    with open(path, "r") as f:
        content=f.readlines()
    n=len(content)
    r=0
    ret=[]

    while r<n:
        if not ">" in content[r]:
            r-=1
        if "id" in content[r]:
            ID=content[r].replace("\n", "").split(" ")[0].split("_")[-1]
        else:
            ID=content[r].replace("\n", "").split(" ")[-1]

        r+=1
        SEQ=content[r].replace("\n", "").upper()
        length=len(SEQ)

        r+=1

        ENE=-9999.0

        STRU=[content[r].replace("\n", "")]
        slabel = [[float(_ == ".") for _ in x] for x in STRU]
        print(len(STRU[0]), length)
        assert len(STRU[0])==length

        LABEL=[]
        with open(path+".predict.files/{}.predict.txt".format(ID), "r") as f:
            fgt=f.readlines()
        fgt=[float(x.replace("\n", "")) for x in fgt]
        if has_label and r+1<len(content) and len(content[r+1])>2:
            while length>0:
                length-=1
                r+=1
                LABEL.append(float(content[r].replace("\n", "").split(" ")[-1]))
            assert len(LABEL)==len(SEQ)
        else:
            LABEL=fgt#[float(x==".") for x in STRU[0]]
        r+=2
        seq0=SEQ
        SEQ=[Dict[x] for x in SEQ]

        STRU=[[Labeldict[_] for _ in x] for x in STRU]
        assert len(STRU)==1
        STRU=STRU[0]
        ret.append([ID, np.array(SEQ), np.array(STRU), np.array(LABEL), np.array(fgt), seq0])
        print(len(ret))
        #print(ID)
    return ret




class TrainDataSet():
    def __init__(self, path):
        self.path=path
        self.data=parse_file(self.path)

    def generate_train_data(self):
        def reader():
            for idx in range(len(self.data)):
                data=self.data[idx]
                id, seq, stru, label, fgt, SEQ=data
                yield id, seq, stru, label, fgt, SEQ
        return reader



def write_res(res, path):
    for x in res:
        p=os.path.join(path, "{}.predict.txt".format(x[0]))
        with open(p, "w") as f:
            for l in x[1]:
                f.writelines("{}\n".format(l))


if __name__=="__main__":
    temp=TrainDataSet("dev.txt")
    temp.generate_train_data(0)

