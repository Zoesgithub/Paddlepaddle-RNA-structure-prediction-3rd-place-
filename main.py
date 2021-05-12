import argparse
import importlib
from loguru import logger
from utils import TrainDataSet, write_res
import numpy as np
import os
import re
import copy
import paddle.fluid as fluid

def check_path(path, eval=False):
    if not os.path.exists(path):
        os.mkdir(path)
        return None
    List=os.listdir(path)
    if List:
        Files=[]
        for File in List:
            name=re.split('-', File)[-1]
            try:
                Files.append(int(name))
            except:
                if  eval and "best" in name:
                    return "model.ckpt-best"
                continue
        if len(Files)>0:
            Files.sort()
            return path+"model.ckpt-{}".format(Files[-1])
        else:
            return None
    else:
        return None

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="The path to the config file")
    args=parser.parse_args()

    config=importlib.import_module(args.config)
    config=config.config
    model=config["model"]

    logger.add(os.path.join(config['log_path'], "train_log"))
    if not os.path.exists(os.path.join(config["log_path"], "model/")):
        os.mkdir(os.path.join(config["log_path"], "model"))

    logger.info("The config file is {}".format(config))


    traindata=TrainDataSet(config["train_path"])
    testdata=TrainDataSet(config["test_path"])
    devdata=TrainDataSet(config["dev_path"])
    traindata=fluid.io.batch(fluid.io.shuffle(traindata.generate_train_data(), buf_size=500), batch_size=1)
    testdata=fluid.io.batch(fluid.io.shuffle(testdata.generate_train_data(), buf_size=500), batch_size=1)
    devdata=fluid.io.batch(fluid.io.shuffle(devdata.generate_train_data(), buf_size=500), batch_size=1)
    #print(traindata())
    #logger.info("Train data size {} Test data size {}".format(len(traindata), len(testdata)))
    EPOCHS=config["epochs"]
    Model=model
    Pred=[]
    Gt=[]
    for d in range(config["num_models"]):
        model=Model#copy.deepcopy(Model)
        for epoch in range(EPOCHS):
            loss=[]
            for i, data in enumerate(traindata()):
                loss_=model.train_onestep(data)
                #print(loss_)
                loss.append(np.array(loss_)[0])
                if i%100==0:
                    print(i)

                    print("Eval ... ")
                    acc, rmsd, ret, gt=model.Eval(testdata())
                    print(acc, rmsd)
                    dacc, drmsd, dret, dgt=model.Eval(devdata())
                    logger.info("EPOCH {} : ACC={} RMSD={} loss={}".format(epoch, dacc, drmsd, np.mean(loss)))

                    Pred=ret#.append(ret)
                    Gt=gt#.append(gt)
                    pred=[np.array(x[1]) for x in Pred]
                    gt=[np.array(x[1]) for x in Gt]
                    print([x.shape for x in pred], [x.shape for x in gt])
                    #print(pred.shape, gt.shape)
                    rmsd=pred#[0]
                    res=list(zip(gt, pred))
                    rmsd=np.mean([np.sqrt(((x[0]-x[1])**2).mean()) for x in res])

                    logger.info("Round {} : RMSD={} ".format(d, rmsd))
                    path=config["respath"]+"_{}_{}".format(epoch, i//100)#os.path.join(config["log_path"], "res{}/".format(d))
                    os.mkdir(path)
                    write_res(Pred,path)


if __name__=="__main__":
    main()


