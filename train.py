import os 
import json
import time
import math
import numpy as np
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from ConvLSTM import Seq2Seq
import pickle #why didn't I use torch for pickling????

#RCWO

#I had all this written as a main file... I hate main files... so I rewrote it again

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

args = {'data_path':'./data',
        'model_name':'clouds4',
        'seq_len' :8,
        'horizon' : 8,
        'h_channels': 64,
        'in_channels':1,
        'out_channels':1,
        #'seq_len':8, #how many to use for known input
        #'horizon':4, #how many to forecast out
        'num_layers': 4,
        'kernel_size':3, #training stuff below:
        'num_epochs':10,
        'result_path': './results',
        'use_teacher_forcing': False,
        'batch_size': 2,
        'lr': .002,
        'device':device}
        
args = dotdict(args)


#load the data:
fnames = os.listdir(args.data_path)

totims = len(fnames)
tr_ex = np.floor(.8*totims).astype(int)

tr_ims = []
te_ims = []
for i in range(totims):
    try:
        f = open(os.path.join(args.data_path,fnames[i]),'rb')
        im = pickle.load(f)
        f.close() #I forgot this bloody line when I created them... problematic?
        if i<=tr_ex:
            tr_ims.append(torch.tensor(im[:,None,:,:]))
        else:
            te_ims.append(torch.tensor(im[:,None,:,:]))
    except:
        print('issue with loading',i)

tr_ex = len(tr_ims)

for q in [1]: #this is where the main started
    S = Seq2Seq(args).to(args.device)
    optimizer = optim.SGD(S.parameters(), lr=args.lr, momentum=0.9)
    
    writer = SummaryWriter(log_dir=args.result_path)  # initialize tensorboard writer
    print(args.use_teacher_forcing)
    if args.use_teacher_forcing:
        tf = 1.0
        print('using teacher forcing') 
    else:
        tf = 0 
        print('not using teacher forcing')
                
    print(tf)
    

    # Decay LR by a factor of 0.1 every 7 epochs
    #exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    bsz = args.batch_size
    k = 0
    lall = 0
    losses = []
    
    for j in range(10):#epochs
        for i in range(tr_ex):
            
            im = tr_ims[i]
            #extract cloud portion:
            tr = (im[0:args.seq_len,:,:,:]==2).to(torch.float32).to(args.device)
            teim = im[args.seq_len:(args.seq_len+args.horizon),:,:,:] #need this in a bit
            te = (teim==2).to(torch.float32).to(args.device)
            
            out = S(tr,te,tf*1/(j+1))
            #don't penalize what happens off earth: vvvv
            loss = torch.mean(( (out[0]-te)*((teim!=3).to(torch.float32).to(args.device)) )**2)
            
            lall = lall+loss
            k+=1
            if k % bsz == 0:
                lall = lall/bsz
                print(j,i,lall)
                lall.backward()
                writer.add_scalar('training loss',lall.item(), j*tr_ex + i)
                losses.append(lall.item())
                optimizer.step()
                optimizer.zero_grad()
                lall=0
                k=0
                
        torch.save(losses,os.path.join(args.result_path,args.model_name+'_losses.pckl'))
        torch.save(S.state_dict(), os.path.join(args.result_path, args.model_name+'ep_' + str(j) +'.pth'))
    
    writer.add_graph(S,(tr,te))
    

 
    
