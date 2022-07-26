import torch.nn as nn
import torch
import argparse
#from moving_mnist import MovingMNIST
import torch.optim as optim
from torch.optim import lr_scheduler

import os

#RCWO

class ConvLSTMCell(nn.Module):

    def __init__(self, in_channels, h_channels, kernel_size, device):
        super(ConvLSTMCell, self).__init__()

        self.h_channels = h_channels
        padding = kernel_size // 2, kernel_size // 2
        self.conv = nn.Conv2d(in_channels=in_channels + h_channels,
                              out_channels=4 * h_channels,
                              kernel_size=kernel_size,
                              padding=padding,
                              bias=True)
        self.device = device
    
    def forward(self, input_data, prev_state):
        h_prev, c_prev = prev_state
        combined = torch.cat((input_data, h_prev), dim=1)  # concatenate along channel axis

        combined_output = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_output, self.h_channels, dim=1)

        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_cur = f * c_prev + i * g
        h_cur = o * torch.tanh(c_cur)

        return h_cur, c_cur

    def init_hidden(self, batch_size, image_size):
        """ initialize the first hidden state as zeros """
        height, width = image_size
        return (torch.zeros(batch_size, self.h_channels, height, width, device=self.device),
                torch.zeros(batch_size, self.h_channels, height, width, device=self.device))

    
    
    
class ConvLSTM(nn.Module):
    
    def __init__(
        self, 
        in_channels, 
        h_channels, 
        num_layers,
        kernel_size, 
        device):
        
        super(ConvLSTM, self).__init__()
        
        self.in_channels = in_channels
        self.num_layer = num_layers
        
        layer_list = []
        for i in range(num_layers):
            cur_in_channels = in_channels if i == 0 else h_channels[i - 1]
            layer_list.append(ConvLSTMCell(in_channels=cur_in_channels,
                                           h_channels=h_channels[i],
                                           kernel_size=kernel_size,
                                           device=device))
            
        self.layer_list = nn.ModuleList(layer_list)
        self.h_channels = h_channels
            
    def forward(self, x, states=None):
                    
        if states is None:
            bsz = x.shape[0]
            imsz = x.shape[-2:]
            hidden_states, cell_states = self.init_hidden(bsz,imsz) 
        else:
            hidden_states, cell_states = states
        
        #hidden_states - list or something
        
        for i, layer in enumerate(self.layer_list):
            if i ==0:
                hidden_states[0],cell_states[0] = layer(x,(hidden_states[0],cell_states[0]))
            else:
                hidden_states[i],cell_states[i] = layer(hidden_states[i-1],(hidden_states[i],cell_states[i]))
        
        return hidden_states, (hidden_states, cell_states)

    def init_hidden(self,bsz,imsz):
        hout = []
        cout = []
        for i, layer in enumerate(self.layer_list):
            hi,ci = layer.init_hidden(1, imsz)
            hout.append(hi)
            cout.append(ci)
        
        return hout,cout
        
#C = ConvLSTM(64,[64]*10,10,3,'cpu')
    
def activation_factory(name):
    """
    Returns the activation layer corresponding to the input activation name.
    Parameters
    ----------
    name : str
        'relu', 'leaky_relu', 'elu', 'sigmoid', or 'tanh'. Adds the corresponding activation function after the
        convolution.
    """
    if name == 'relu':
        return nn.ReLU(inplace=True)
    if name == 'leaky_relu':
        return nn.LeakyReLU(0.2, inplace=True)
    if name == 'elu':
        return nn.ELU(inplace=True)
    if name == 'sigmoid':
        return nn.Sigmoid()
    if name == 'tanh':
        return nn.Tanh()
    if name is None or name == "identity":
        return nn.Identity()

    raise ValueError(f'Activation function `{name}` not yet implemented')

    
def make_conv_block(conv):

    out_channels = conv.out_channels
    modules = [conv]
    modules.append(nn.GroupNorm(16, out_channels))
    modules.append(nn.LeakyReLU(0.2, inplace=True))
    return nn.Sequential(*modules)


class DCGAN64Encoder(nn.Module):
    
    def __init__(self, in_c, out_c):

        super(DCGAN64Encoder, self).__init__()
        
        h_c = out_c // 2 
        self.conv = nn.ModuleList([
            make_conv_block(nn.Conv2d(in_c, h_c, 3, 2, 1)),
            make_conv_block(nn.Conv2d(h_c, h_c, 3, 1, 1)),
            make_conv_block(nn.Conv2d(h_c, out_c, 3, 2, 1))])
        
    def forward(self, x):
        out = x
        for layer in self.conv:
            out = layer(out)
        return out
        

class DCGAN64Decoder(nn.Module):
    
    def __init__(self, in_c, out_c, last_activation='sigmoid'):
        
        super(DCGAN64Decoder, self).__init__()
        
        h_c = in_c // 2
        self.conv = nn.ModuleList([
            make_conv_block(nn.ConvTranspose2d(in_c, h_c, 3, 2, 1, output_padding=1)),
            make_conv_block(nn.ConvTranspose2d(h_c, h_c, 3, 1, 1)),
            nn.ConvTranspose2d(h_c, out_c, 3, 2, 1, output_padding=1)])
            #make_conv_block(nn.ConvTranspose2d(h_c, out_c, 3, 2, 1, output_padding=1))])

        self.last_activation = activation_factory(last_activation)

    def forward(self, x):
        out = x
        for i, layer in enumerate(self.conv):
            out = layer(out)
        return self.last_activation(out)    
    
    
    
class Seq2Seq(nn.Module):
    
    def __init__(self, args):
        
        super(Seq2Seq, self).__init__()
                
        
        self.in_channels = args.in_channels
        self.out_channels = args.out_channels
        self.h_channels = args.h_channels
        self.num_layers = args.num_layers
        self.kernel_size = args.kernel_size
                
        self.seq_len = args.seq_len
        self.horizon = args.horizon
        
        self.frame_encoder = DCGAN64Encoder(self.in_channels, self.h_channels).to(args.device)
        self.frame_decoder = DCGAN64Decoder(self.h_channels, self.out_channels).to(args.device)

        self.model = ConvLSTM(in_channels=self.h_channels, 
                              h_channels=[self.h_channels] * self.num_layers, #THIS IS F'ING COOL
                              num_layers=self.num_layers, 
                              kernel_size=self.kernel_size,
                              device=args.device)
                    
    def forward(self, in_seq, out_seq, tf=0):
        #tf teacher forcing rate
        #rando number < tf -> use ground truth
        #send tf to 0 to stop using gt
        
        #out seq: len(in_seq) + k_pred
        
        #tf defaults to 0 - i.e. not using forcing
        
        next_frames = []
        hidden_states, states = None, None
        
        # encoder:
        enc = self.frame_encoder(in_seq)
        enc2 = self.frame_encoder(out_seq)
        
        for t in range(self.seq_len - 1):
            hidden_states, states = self.model(enc[t][None,:,:,:], states) 
            #no-comprende: why are we not using teacher forcing here?                                   
                                                
        # decoder
        for t in range(self.horizon):
            if torch.rand(1) < tf: #use gt
                if t ==0: 
                    x = enc[-1][None,:,:,:]
                else: #train on previous
                    x = enc2[t-1][None,:,:,:]
            else: #use output of previous
                x = hidden_states[-1] #think this is right
            
            hidden_states, states = self.model(x, states)
            next_frames.append(self.frame_decoder(hidden_states[-1]))
        
        next_frames = torch.stack(next_frames, dim=1) 
        return next_frames
    


#ADD ARGS.DEVICE!!!!!!!


'''
class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

args = {'seq_len' : 10,
        'horizon' : 3,
        'h_channels': 64,
        'in_channels':1,
        'out_channels':1,
        'num_layers': 3,
        'kernel_size':3,
        'device':device}
#so... this only really works with 64
#no idea why....

args = dotdict(args)
S = Seq2Seq(args).to(device)


#fyi: sequences are 20-long, so don't load more than that
train_dataset = MovingMNIST(root='../data/',
                            is_train=True, 
                            seq_len=10,
                            horizon=3)

optimizer = optim.SGD(S.parameters(), lr=0.005, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
#exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)


bsz = 30
k = 0
lall = 0

model_save_dir = './mnist'


for j in range(10):#epochs
    for i in range(len(train_dataset)):
        tr,te = next(iter(train_dataset))
        tr = tr.to(device)
        te = te.to(device)
        out = S(tr,te,1/(j+1))
        loss = torch.mean((out-te)**2)
        
        lall = lall+loss
        k+=1
        if k % bsz == 0:
            lall = lall/bsz
            print(j,i,lall)
            lall.backward()
            optimizer.step()
            optimizer.zero_grad()
            lall=0
            k=0
    torch.save(S.state_dict(), os.path.join(model_save_dir, 'ep_' + str(j) +'.pth'))
'''




