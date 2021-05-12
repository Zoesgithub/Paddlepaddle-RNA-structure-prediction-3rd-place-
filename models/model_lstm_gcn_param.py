import math, random
import RNA

import numpy as np
import paddle
import paddle.fluid as fluid
from paddle.fluid.dygraph import Layer
from .egsage import egsage
from functools import partial
def load_params(path):
    with open(path, "r") as f:
        content=f.readlines()
    ret=[]
    params=[]
    ret.append(content[:4])
    ret.append(content[11:])
    for i in range(4, 11):
        c=content[i].split("  ")[1:-2]
        c=[x for x in c if len(x)>0]
        params.extend([float(x) for x in c])
    return ret, params

def write_params(path, others, params):
    with open(path, "w") as f:
        for line in others[0]:
            f.writelines(line)
        for i in range(7):
            line="  {:.2f}  {:.2f}  {:.2f}  {:.2f}  {:.2f}  {:.2f}  {:.2f}    /* CG */\n".format(params[i*7], params[i*7+1], params[i*7+2], params[i*7+3], params[i*7+4], params[i*7+5], params[i*7+6])
            f.writelines(line)
        for line in others[1]:
            f.writelines(line)

def store_structure(s, data):
    if s:
        data.append(np.array([x!="." for x in s]))
        #data.append(s)
class Model(Layer):
    def __init__(self, lr
                 ):
        super(Model, self).__init__()
        self.seq_embedding = partial(paddle.fluid.embedding, size=(4, 128), is_sparse=False)
        self.bra_embedding=partial(paddle.fluid.embedding, size=(3, 128), is_sparse=False)
        self.net1=partial(paddle.fluid.layers.dynamic_lstm, size=128*16, use_peepholes=False)
        self.net2=partial(paddle.fluid.layers.dynamic_lstm, size=128*16, use_peepholes=False, is_reverse=True)
        self.inp_fc=partial(paddle.fluid.layers.fc, size=128, act="relu")
        self.seq_fc=partial(paddle.fluid.layers.fc, size=128, act="relu")
        self.fc=partial(paddle.fluid.layers.fc, size=128*16, act="relu")

        self.out_linear1=partial(paddle.fluid.layers.fc, size=128, act="relu")
        self.out_linear2=partial(paddle.fluid.layers.fc, size=28)

        self.gcn1=egsage(256*4, 256*4, 1)
        self.gcn2 = egsage(256*4, 256*4, 1)
        self.gcn3 = egsage(256*4, 256*4, 1)
        #self.gcn4 = egsage(256*4, 256*4, 1)

        self.sig=paddle.fluid.layers.sigmoid
        self.soft=partial(paddle.fluid.layers.softmax, axis=-1)

        #for train
        trainer_count = fluid.dygraph.parallel.Env().nranks
        self.place= fluid.CUDAPlace(fluid.dygraph.parallel.Env().dev_id) if trainer_count > 1 else fluid.CUDAPlace(0)
        self.exe = fluid.Executor(self.place)
        #self.exe_test=fluid.Executor(self.place)
        self.startup_program = fluid.default_startup_program()
        self.main_program=fluid.default_main_program()#fluid.Program()
        self.optimizer=fluid.optimizer.Adam(learning_rate=lr)
        self.seq=fluid.data(name="seq", shape=[None], dtype="int64", lod_level=1)
        self.dot=fluid.data(name="dot", shape=[None], dtype="int64", lod_level=1)
        self.edge_attr=fluid.data(name="edge_attr", shape=[1, None], dtype="float32", lod_level=1)
        self.edge_index=fluid.data(name="edge_index", shape=[2, None], dtype="int64", lod_level=1)

        self.y=fluid.data(name="label", shape=[None], dtype="float32")
        self.fgt=fluid.data(name="fgt", shape=[None, 1], dtype="float32")
        self.dtv=fluid.data(name="dtv", shape=[28], dtype="float32")
        self.prediction=self.forward(self.seq, self.dot, self.edge_index, self.edge_attr, self.fgt)
        #self.loss=fluid.layers.mean(fluid.layers.mse_loss(self.prediction, label=self.y)**0.5)
        self.feeder=paddle.fluid.DataFeeder(place=self.place, feed_list=[self.seq, self.dot,self.edge_attr, self.edge_index,  self.y, self.fgt, self.dtv])
        self.loss=fluid.layers.mean(sum([self.dtv[i]*self.prediction[i] for i in range(28)]))
        self.optimizer.minimize(self.loss)
        self.test_program=self.main_program.clone(for_test=True)
        #self.test_feeder=paddle.fluid.DataFeeder(place=self.place, feed_list=[self.seq, self.dot,self.edge_attr, self.edge_index])
        self.exe.run(self.startup_program)
        self.ret, self.params0=load_params("rna_turner2004.par0")
        self.params0=np.array(self.params0)
        pp=[]
        for i in range(7):
            for j in range(i, 7):
                pp.append(self.params0[i*7+j])
        self.params0=np.array(pp)
        #self.exe_test.run(self.test_program)
        
    def gen_edge(self, dot):
        edge=[]
        edge_attr=[]
        stack=[]
        for i,x in enumerate(dot):
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
        edge=np.transpose(np.array(edge), (1, 0))
        edge_attr=np.array(edge_attr).reshape(-1,1)
        return edge, edge_attr

    def forward(self, seq, dot, edge, edge_attr, fgt):
        emb_seq=self.seq_embedding(seq)
        #emb_seq=paddle.fluid.layers.concat(input=[emb_seq, fgt], axis=1)


        emb_dot=(self.bra_embedding(dot))

        emb_dot=self.inp_fc(emb_dot)
        emb_seq=self.seq_fc(emb_seq)

        emb=self.fc(paddle.fluid.layers.concat(input=[emb_seq, emb_dot], axis=1))#.permute(1, 0, 2)

        emb1, _=self.net1(emb)
        emb2, _=self.net2(emb)
        emb=paddle.fluid.layers.concat(input=[emb1, emb2], axis=1)
        print(emb.shape, emb1.shape, emb2.shape)
        emb_g=self.gcn1(emb, edge_attr, edge)
        emb_g = self.gcn2(emb_g, edge_attr, edge)
        emb_g = self.gcn3(emb_g, edge_attr, edge)
        #emb_g = self.gcn4(emb_g, edge_attr, edge)
        emb=paddle.fluid.layers.concat([emb, emb_g], -1)
        emb=paddle.fluid.layers.reduce_mean(emb, dim=0, keep_dim=True)
        out=self.out_linear2(self.out_linear1(emb))#.permute(1, 0, 2)
        out=out
        return fluid.layers.reshape(out, [-1])*10.0

    def trans_params(self, params):
        ret=np.zeros([7,7])
        idx=0
        for i in range(7):
            for j in range(i, 7):
                ret[i][j]=params[idx]
                ret[j][i]=params[idx]
                idx+=1
        return ret.reshape(-1).tolist()
    def get_out(self,ret, params, seq, size=100):
        #print(seq)
        params=self.trans_params(params)
        write_params("rna_turner2004.par", ret, params)
        RNA.read_parameter_file("rna_turner2004.par")
        md=RNA.md()
        md.dangles=2
        md.uniq_ML=1
        fc=RNA.fold_compound(seq, md)
        _, mfe=fc.mfe()
        fc.exp_params_rescale(mfe)
        fc.pf()
        ss=fc.bpp()#list()
        ss=np.array(ss)
        ss=ss+np.transpose(ss)

        #i=fc.pbacktrack(size, store_structure, ss)
        return 1-np.array(ss).sum(1)[1:]

    def train_onestep(self, data):
        #print(data)
        id, seq, dot, label, fgt, SEQ=data[0]
        seq=seq.reshape(-1)
        dot=dot.reshape(-1)
        label=label.reshape(-1)
        fgt=fgt.reshape(-1, 1)
        edge, edge_attr=self.gen_edge(dot)
        pred, =self.exe.run(self.test_program, feed=self.feeder.feed([[seq, dot, edge_attr, edge, label, fgt, np.zeros(28)]]), fetch_list=[ self.prediction.name], return_numpy=False)
        pred=np.array(pred)+self.params0

        s0=self.get_out(self.ret, pred, SEQ)
        l0=np.mean((label-s0)**2)**0.5
        dtv=[]
        avgsi=[]
        for i in range(28):
            ipred=np.copy(pred)
            ipred[i]=ipred[i]+1
            si=self.get_out(self.ret, ipred, SEQ)
            li=np.mean((label-si)**2)**0.5
            avgsi.append(li)
            dtv.append(np.sign(li-l0))
        print(l0, sum(avgsi)/28, max(avgsi), min(avgsi))
        loss, pred=self.exe.run(self.main_program, feed=self.feeder.feed([[seq, dot, edge_attr, edge, label, fgt, dtv]]), fetch_list=[self.loss.name, self.prediction.name], return_numpy=False)
        #print(pred)
        return loss

    def Eval(self, dataloader):
        acc = 0.0
        size = 0.0
        rmsd = 0.0
        ret = []
        gt = []
        RSize = []
        for data in dataloader:
            id, seq, dot, label, fgt, SEQ=data[0]
            seq=seq.reshape(-1)
            dot=dot.reshape(-1)
            label=label.reshape(-1)
            fgt=fgt.reshape(-1,1 )
            edge, edge_attr=self.gen_edge(dot)
            y=label
            pred, =self.exe.run(self.test_program, feed=self.feeder.feed([[seq, dot, edge_attr, edge, y, fgt, np.zeros(28)]]), fetch_list=[ self.prediction.name], return_numpy=False)
            pred=np.array(pred)+self.params0
            pred=self.get_out(self.ret, pred, SEQ, size=2000)
            #print(pred)
            #exit()
            pred=pred.reshape(1, -1)
            y=np.array(y).reshape(1, -1)

            #for i, d in enumerate(id):
            #    print(d, id)
            ret.append([id, pred[0]])
            gt.append([id, y[0]])
            rmsd += np.sqrt(((pred - y) ** 2).mean(1)).sum()
            size +=1.0

        acc = 0
        rmsd = rmsd / size
        return acc, rmsd, ret, gt

