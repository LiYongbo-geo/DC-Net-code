import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

class GravEncoder(nn.Module):
    '''Encode a gravity field to a the 1st layer of 3-D density model. 
    '''
    def __init__(self, dim=3, *args, **kwargs):
        super(GravEncoder, self).__init__()
        self._name = 'GravEncoder'
        self.dim = dim
        self.pre_layers = 6
        self.down_layers =5 
        self.up_layers = 5
        self.post_layers = 3
        self.pre_conv_layers = nn.ModuleList()
        self.pre_norm_layers = nn.ModuleList()
        self.pre_adj_conv_layers = nn.ModuleList()
        self.pre_adj_norm_layers = nn.ModuleList()
        self.down_conv_layers = nn.ModuleList()
        self.down_norm_layers = nn.ModuleList()
        self.down_adj_layers = nn.ModuleList()
        self.up_conv_layers = nn.ModuleList()
        self.up_norm_layers = nn.ModuleList()
        self.up_adj_conv_layers = nn.ModuleList()
        self.up_adj_norm_layers = nn.ModuleList()
        self.post_conv_layers = nn.ModuleList()
        self.post_norm_layers = nn.ModuleList()
        # preprocess
        c_in = 1  #输入通道数
        c_out = 2  #输出通道数
        for i in range(self.pre_layers):
            self.pre_conv_layers.append(nn.Conv2d(c_in,c_out,3,padding=1,bias=False)) #卷积层，
            self.pre_norm_layers.append(nn.BatchNorm2d(c_out))  #数据归一化处理
            self.pre_adj_conv_layers.append(nn.Conv2d(c_out,c_out,3,padding=1,bias=False))
            self.pre_adj_norm_layers.append(nn.BatchNorm2d(c_out)) 
            c_in = c_out
            c_out = 2**(i+1)
        # downsampling network
        #c_in = c_out
        for i in range(self.down_layers):
            self.down_conv_layers.append(nn.Conv2d(c_in,c_in,3,padding=1,bias=False))
            self.down_norm_layers.append(nn.BatchNorm2d(c_in))
            self.down_adj_layers.append(nn.Conv2d(c_in,c_out,3,padding=1,bias=False))
            c_in = c_out
            c_out = c_out*2
        c_out = int(c_in/2)
        # upsampling network
        for i in range(self.up_layers):
            self.up_conv_layers.append(nn.ConvTranspose2d(c_in,c_out,2,stride=2,bias=False))
            self.up_norm_layers.append(nn.BatchNorm2d(c_out))
            self.up_adj_conv_layers.append(nn.Conv2d(c_out,c_out,3,padding=1,bias=False))
            self.up_adj_norm_layers.append(nn.BatchNorm2d(c_out))
            c_in = 2*c_out
            c_out = int(c_out/2)
        # postprocess
        for i in range(self.post_layers):
            self.post_conv_layers.append(nn.Conv2d(c_in,c_in,3,padding=1,bias=False))
            self.post_norm_layers.append(nn.BatchNorm2d(c_in))
        self.final_conv_layers_1 = nn.Conv2d(c_in,1,3,padding=1)

    def forward(self, data_input):
        ''' calculate gravity model from observed field
            Args:
                data_input (Tensor): observed gravity field, with shape (nbatch,1,ny,nx)
            Returns:
                out (Tensor): inversion result, with shape (nbatch,nz,ny,nx)
        '''
        tmp = data_input
        out = []
        out.append(tmp)
        # construct 32x128x128 data
        for i in range(self.pre_layers):
            tmp = self.pre_conv_layers[i](tmp)
            tmp = F.relu(self.pre_norm_layers[i](tmp))
            tmp = self.pre_adj_conv_layers[i](tmp)
            tmp = F.relu(self.pre_adj_norm_layers[i](tmp))
            out.append(tmp) 
        #tmp = torch.cat(out,dim=1)
        # down smaple
        out2 = []
        
        for i in range(self.down_layers):
            out2.append(tmp)
            tmp = self.down_conv_layers[i](tmp)
            tmp = F.max_pool2d(F.relu(self.down_norm_layers[i](tmp)),2)
            tmp = self.down_adj_layers[i](tmp)
        # upsampling
        for i in range(self.up_layers):
            tmp = self.up_conv_layers[i](tmp)
            tmp = F.relu(self.up_norm_layers[i](tmp))
            tmp = self.up_adj_conv_layers[i](tmp)
            tmp = F.relu(self.up_adj_norm_layers[i](tmp))
            try:
                tmp = torch.cat([tmp,out2.pop()],dim=1)
            except IndexError:
                pass
        for i in range(self.post_layers):
            tmp = self.post_conv_layers[i](tmp)
            tmp = F.relu(self.post_norm_layers[i](tmp))    
        tmp = self.final_conv_layers_1(tmp)
        return tmp