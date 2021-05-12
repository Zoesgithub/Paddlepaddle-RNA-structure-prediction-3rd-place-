from models.model_lstm_gcn_param import Model 
import torch.nn as nn
config={
        "model":Model(1e-3),
        "train_path":"data/dev/train_full.txt",
        #"train_path":"data/dev/temp.txt",
        "test_path":"data/stageB/B_board_112_seqs.txt",
        #"test_path":"data/dev/dev.txt",
        "dev_path":"data/dev/dev.txt",
        #"dev_path":"data/dev/temp.txt",
        
        "epochs":15,
        "log_path":"tasks/rl/",
        "use_label":True,
        "num_workers":1,
        "cut":True,
        "num_models":1,
        "respath":"tasks/rl/res19"

        }
