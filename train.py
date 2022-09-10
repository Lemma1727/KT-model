import os
import argparse
import json
import torch

from torch.optim import Adam

from data_loader.ASSIST2009 import assist2009
from data_loader.ASSIST2015 import assist2015
from models.dkt import DKT
from models.dkvmn import DKVMN

def main(model_name, data):
    with open("config.json") as f:
        config = json.load(f)
        data_config = config['data_config']
        train_config = config['train_config']
        model_config = config['model_config'][model_name]
    
    batch_size = data_config['batch_size']
    train_ratio = data_config['train_ratio']
    seq_len = data_config['seq_len']
    
    epochs = train_config['epochs']
    lr = train_config['lr']
    
    if data == "ASSIST2009":
        train_loader, val_loader, num_q = assist2009(train_ratio, batch_size, seq_len)
    elif data == "ASSIST2015":
        train_loader, val_loader, num_q = assist2015(train_ratio, batch_size, seq_len)
        
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if model_name == "DKT":
        model = DKT(num_q, **model_config).to(device)
    elif model_name == "DKVMN":
        model = DKVMN(num_q, **model_config).to(device)
    else:
        print("model name was worng")
        return
    
    optimizer = Adam(model.parameters(), lr)
    
    aucs, loss_means = model.run_train(train_loader, val_loader, epochs, optimizer)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type = str,
        default = "DKT"
    )
    parser.add_argument(
        "--data",
        type = str,
        default = "ASSIST2009"
    )
    args = parser.parse_args()
    
    main(args.model_name, args.data)
    