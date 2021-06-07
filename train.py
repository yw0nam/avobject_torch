import numpy as np
import sys
import time
import os
import argparse
import pdb
import glob
import torch
from avobject import avobject_model
from Data_generator import Datagen
from torch.utils.data import DataLoader


parser = argparse.ArgumentParser(description = "TrainArgs");

## Data loader
parser.add_argument('--batch_size', type=int, default=8, help='')
## Training details
parser.add_argument('--max_epoch', type=int, default=100, help='Maximum number of epochs');

## Model definition
parser.add_argument('--nOut', type=int,  default=1024, help='Embedding size in the last FC layer');
parser.add_argument('--n_neg', type=int,  default=4, help='negative sample number');

## Learning rates
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate');
parser.add_argument("--lr_decay", type=float, default=0.95, help='Learning rate decay every epoch');

## Load and save
parser.add_argument('--initial_model', type=str, default="", help='Initial model weights');
parser.add_argument('--save_path',     type=str, default="./data/exp01", help='Path for model and logs');

## Training and test data
parser.add_argument('--train_list', type=str, default="LRS3/dev.txt", help='');
parser.add_argument('--verify_list', type=str, default="LRS3/test.txt", help='');


args = parser.parse_args();

# ==================== MAKE DIRECTORIES ====================

model_save_path     = args.save_path+"/model"
result_save_path    = args.save_path+"/result"

if not(os.path.exists(model_save_path)):
    os.makedirs(model_save_path)

if not(os.path.exists(result_save_path)):
    os.makedirs(result_save_path)

# ==================== LOAD MODEL ====================

s = avobject_model(learning_rate=args.lr, nOut=args.nOut, n_neg=args.n_neg)

# ==================== EVALUATE LIST ====================

it = 1;

scorefile = open(result_save_path+"/scores.txt", "a+");

for items in vars(args):
    print(items, vars(args)[items]);
    scorefile.write('%s %s\n'%(items, vars(args)[items]));
scorefile.flush()

# ==================== LOAD MODEL PARAMS ====================

modelfiles = glob.glob('%s/model0*.model'%model_save_path)
modelfiles.sort()

if len(modelfiles) >= 1:
    s.loadParameters(modelfiles[-1]);
    print("Model %s loaded from previous state!"%modelfiles[-1]);
    it = int(os.path.splitext(os.path.basename(modelfiles[-1]))[0][5:]) + 1
elif(args.initial_model != ""):
    s.loadParameters(args.initial_model);
    print("Model %s loaded!"%args.initial_model);

for ii in range(0,it-1):
    clr = s.updateLearningRate(args.lr_decay) 
    
# ==================== LOAD DATA LIST ====================

print('Reading data ...')

train_dataset = Datagen(args.train_list)
val_dataset = Datagen(args.verify_list)
trainLoader = DataLoader(train_dataset, batch_size=args.batch_size, 
                         shuffle=True, pin_memory=True, drop_last=True)
valLoader = DataLoader(val_dataset, batch_size=args.batch_size, drop_last=True, 
                       shuffle=True, pin_memory=True)
print('Reading done.')

# ==================== CHECK SPK ====================

clr = s.updateLearningRate(1)

while(1):   
    print(time.strftime("%Y-%m-%d %H:%M:%S"), it, "Start Iteration");

    loss = s.train_network(trainLoader, evalmode=False);
    valloss = s.train_network(valLoader,   evalmode=True);


    print(time.strftime("%Y-%m-%d %H:%M:%S"), "%s: IT %d, LR %f,TLOSS %f, VLOSS %f\n"%(args.save_path, it, max(clr), loss, valloss));
    scorefile.write("IT %d, LR %f, TLOSS %f, VLOSS %f\n"%(it, max(clr), loss, valloss));
    scorefile.flush()

    # ==================== SAVE MODEL ====================

    clr = s.updateLearningRate(args.lr_decay) 

    print(time.strftime("%Y-%m-%d %H:%M:%S"), "Saving model %d" % it)
    s.saveParameters(model_save_path+"/model%09d.model"%it);

    if it >= args.max_epoch:
        quit();

    it+=1;
    print("");

scorefile.close();