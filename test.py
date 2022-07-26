import os
import numpy as np
import argparse
import torch
import pickle
from torch.utils.data import DataLoader
from skimage.metrics import structural_similarity as compare_ssim
from ConvLSTM import Seq2Seq

#RCWO


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

args = {'data_path':'./data',
        'result_path':'./results',
        'model_name':'clouds4',
        'seq_len' : 8,
        'horizon' : 8,
        'h_channels': 64,
        'in_channels':1,
        'out_channels':1,
        'num_layers': 4,
        'kernel_size':3, #training stuff below:
        'device':device}
        
args = dotdict(args)

modelfile = './clouds3final.pth' #args.model_name+'ep_9.pth'


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
        
        
te_ex = len(te_ims)



for q in [1]:
    
    num_samples = 0
    avg_mse = 0
    avg_ssim = 0
    frame_mse = [0] * args.horizon
    frame_ssim = [0] * args.horizon
 
    ground_truth, prediction = [], []  # these are for visualization

    model = Seq2Seq(args)
    model.load_state_dict(torch.load(os.path.join(args.result_path, modelfile)))
    model.to(device)   
    model.eval()
    with torch.no_grad():
        for i in range(te_ex):
            im = te_ims[i]
            #extract cloud portion:
            #this is bad nomenclature - tr=input seq, te=side wanting to forecast
            tr = (im[0:args.seq_len,:,:,:]==2).to(torch.float32).to(args.device)
            teim = im[args.seq_len:(args.seq_len+args.horizon),:,:,:] #need this in a bit
            te = (teim==2).to(torch.float32).to(args.device)
            
            tr = tr.to(device)
            te = te.to(device)
            out = model(tr,te,0)[0] #we aren't really using te at all here, it's just input
            
            out = out*((teim!=3).to(torch.float32).to(device)) #not care about what's happening off planet
            
            frames = te
            num_samples += 1 #ok
            
            frames = frames.detach().cpu().numpy()
            out = out.detach().cpu().numpy()
            
            #frames = frames[:, -args.horizon:, ...] #this shouldn't be necessary
            #out = out[:, -args.horizon:, ...]
             
            if num_samples < 100:
                ground_truth.append(frames)
                prediction.append(out)

            for i in range(args.horizon):
    
                frame_i = frames[i, ...]
                out_i = out[i, ...]
                mse = np.square(frame_i - out_i).sum()

                ssim = 0
                ssim += compare_ssim(frame_i[0], out_i[0])
                
                frame_mse[i] += mse
                frame_ssim[i] += ssim
                avg_mse += mse
                avg_ssim += ssim
                                
    ground_truth = np.concatenate(ground_truth)
    prediction = np.concatenate(prediction)
            
    avg_mse = avg_mse / (num_samples * args.horizon)
    avg_ssim = avg_ssim / (num_samples * args.horizon)
    
    print('mse: {:.4f}, ssim: {:.4f}'.format(avg_mse, avg_ssim))
    for i in range(args.horizon):
        print('frame {} - mse: {:.4f}, ssim: {:.4f}'.format(i + args.seq_len + 1,
                                                            frame_mse[i] / num_samples,
                                                            frame_ssim[i] / num_samples))
        
    np.savez_compressed(
        os.path.join('{}/{}_{}_prediction.npz'.format(args.result_path, args.model_name, args.data_source)),
        ground_truth=ground_truth,
        prediction=prediction)


