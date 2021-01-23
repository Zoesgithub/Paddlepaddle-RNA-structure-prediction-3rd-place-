import argparse
import importlib
from loguru import logger
from utils import TrainDataSet, PadCollate
from torch.utils.data import DataLoader
import numpy as np
import torch
import os
import re

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
    model=config["model"].cuda()

    logger.add(os.path.join(config['log_path'], "train_log"))
    if not os.path.exists(os.path.join(config["log_path"], "model/")):
        os.mkdir(os.path.join(config["log_path"], "model"))
    else:
        path=check_path(os.path.join(config["log_path"], "model/"))
        if path:
            logger.info("Loading model from {}".format(path))
            checkpoint = torch.load(path)

            model.load_state_dict(checkpoint['param'])

    logger.info("The config file is {}".format(config))


    traindata=TrainDataSet(config["train_path"], cut=config["cut"], aug=False)
    testdata=TrainDataSet(config["test_path"], aug=False)
    traindata=DataLoader(traindata, batch_size=128, shuffle=True, num_workers=config["num_workers"])#, collate_fn=PadCollate(dim=0))
    testdata=DataLoader(testdata, batch_size=128, shuffle=False, num_workers=1)#, collate_fn=PadCollate(dim=0))

    logger.info("Train data size {} Test data size {}".format(len(traindata), len(testdata)))
    EPOCHS=config["epochs"]
    for epoch in range(EPOCHS):
        loss=[]
        for i, data in enumerate(traindata):
            loss_=model.train_onestep(data)
            loss.append(loss_.detach().cpu().numpy())
        print("Eval ... ")
        acc, rmsd, ret=model.Eval(testdata)
        logger.info("EPOCH {} : ACC={} RMSD={} loss={}".format(epoch, acc, rmsd, np.mean(loss)))
        torch.save({"param": model.state_dict()}, os.path.join(config["log_path"], "model/model.ckpt-{}".format(epoch))) 
        if epoch%10==9:
            for param_group in model.optimizer.param_groups:
                param_group["lr"]/=10.0
                lr=param_group["lr"]
            logger.info("Adjust lr {}".format(lr))

if __name__=="__main__":
    main()


