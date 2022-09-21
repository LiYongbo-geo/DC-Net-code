import torch
from torch import nn
import torch.nn.functional as F

class GravDiscriminator(nn.Module):
    ''' Discriminate whether a density model is possibally real. Use 3-d convolutional
        layer.
    Args:
        region_size (int): We doesn't look at the whole density model, instead, we
            look at small regions of it. 'region_size' specify the dimension of this
            region in the unit of number of cells.
        region_stride (int): We use the region as a slide window, 'region_stride' is
            the sliding stride.
        conv_size (int): kernel size of the 3-d convolutional layer.
        conv_stride (int): stride of the 3-d convolutional layer.
    '''
    def __init__(self,
                 region_size=8,
                 region_stride=4,
                 conv_size=3,
                 conv_stride=1):
        super(GravDiscriminator, self).__init__()
        self._name = 'GravDiscriminator'
        self.region_size = region_size
        self.region_stride = region_stride
        self.conv_size = conv_size
        self.conv_stride = conv_stride
        self.n_conv3d = 3
        self.conv3d_x = nn.ModuleList()
        for i in range(self.n_conv3d):
            self.conv3d_x.append(nn.Conv3d(1,1,self.conv_size,
                                         stride=self.conv_stride))
        self.conv3d_y = nn.ModuleList()
        for i in range(self.n_conv3d):
            self.conv3d_y.append(nn.Conv3d(1,1,self.conv_size,
                                         stride=self.conv_stride))
        self.conv3d_z = nn.ModuleList()
        for i in range(self.n_conv3d):
            self.conv3d_z.append(nn.Conv3d(1,1,self.conv_size,
                                         stride=self.conv_stride))
        '''
        self.n_conv2d = 1
        self.conv2d_x = nn.ModuleList()
        for i in range(self.n_conv3d):
            self.conv2d_x.append(nn.Conv2d(32//2**self.n_conv3d,1,self.conv_size,
                                         stride=self.conv_stride))
        self.conv3d_y = nn.ModuleList()
        for i in range(self.n_conv3d):
            self.conv2d_y.append(nn.Conv2d(32//2**self.n_conv3d,1,self.conv_size,
                                         stride=self.conv_stride))
        self.conv3d_z = nn.ModuleList() 
        for i in range(self.n_conv3d):
            self.conv2d_z.append(nn.Conv2d(32//2**self.n_conv3d,1,self.conv_size,
                                         stride=self.conv_stride))
        '''
        self.linear_x = nn.Conv3d(1,1,3)
        self.linear_y = nn.Conv3d(1,1,3)
        self.linear_z = nn.Conv3d(1,1,3)
 
    def forward(self, data_input):
        ''' Discriminate if a density model is possiblly real.
        Args:
            data_input (Tensor): density model, with shape (nbatch,nz,ny,nx)
        Returns:
            res (Tensor): a number indicate how far away the data_input is from
                        reality.
        '''
        out = data_input.unsqueeze(1) * 1000
        gradx = torch.tanh(out[:,:,:,:,1:] - out[:,:,:,:,:-1])
        for tmp_conv3d in self.conv3d_x:
            gradx = F.relu(tmp_conv3d(gradx))
            gradx = F.max_pool3d(gradx,2,ceil_mode=True)
        gradx = torch.sum(self.linear_x(gradx))
        #
        grady = torch.tanh(out[:,:,:,1:,:] - out[:,:,:,:-1,:])
        for tmp_conv3d in self.conv3d_y:
            grady = F.relu(tmp_conv3d(grady))
            grady = F.max_pool3d(grady,2,ceil_mode=True)
        grady = torch.sum(self.linear_y(grady))
        #
        gradz = torch.tanh(out[:,:,1:,:,:] - out[:,:,:-1,:,:])
        for tmp_conv3d in self.conv3d_z:
            gradz = F.relu(tmp_conv3d(gradz))
            gradz = F.max_pool3d(gradz,2,ceil_mode=True)
        gradz = torch.sum(self.linear_z(gradz))
        return (gradx+grady+gradz)/3.
