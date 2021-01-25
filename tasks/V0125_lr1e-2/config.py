from models.model import Model 
import torch.nn as nn
config={
        "model":Model(1e-2),
        "train_path":"data/dev/train_old.txt",
        "test_path":"data/dev/dev.txt",
        
        "epochs":200,
        "log_path":"tasks/V0125_lr1e-2/",
        "use_label":True,
        "num_workers":3,
        "cut":True

        }
