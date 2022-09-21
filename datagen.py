import math
import numpy as np
import torch
from torch.utils.data import IterableDataset,Dataset
import decoder

class DensityDataset(Dataset):
    def __init__(self,
                 max_nlayers=14,
                 max_nintrusion=3,
                 dzyx=(400,500,500),
                 nzyx=(32,64,64),
                 density_range=(0.0,0.0005),
                 anomly_density_range=(0.5,1),
                 seed=None, 
                 length=None,
                 data_dir='./data'):
        super(DensityDataset).__init__()
        self.max_nlayers = max_nlayers
        self.max_nintrusion= max_nintrusion
        self.dzyx = dzyx
        self.nzyx = nzyx
        self.density_range = density_range
        self.anomly_density_range= anomly_density_range
        self.current_idx = 0
        self.seed = seed
        self.length = length
        self.decoder = decoder.GravDecoder(nzyx=self.nzyx,data_dir=data_dir)
        self.positions = np.zeros((*self.nzyx,3),dtype=np.float32)
        xs = np.linspace(-dzyx[2]*(self.nzyx[2]-1)/2,dzyx[2]*(self.nzyx[2]-1)/2,self.nzyx[2]) #密度模型中心点位
        ys = np.linspace(-dzyx[1]*(self.nzyx[1]-1)/2,dzyx[1]*(self.nzyx[1]-1)/2,self.nzyx[1])
        zs = np.linspace(0,dzyx[0]*(self.nzyx[0]-0.5),self.nzyx[0])
        self.positions[:,:,:,0] = np.broadcast_to(xs.reshape(1,1,-1),self.nzyx) #-1,自动计算列数
        self.positions[:,:,:,1] = np.broadcast_to(ys.reshape(1,-1,1),self.nzyx)
        self.positions[:,:,:,2] = np.broadcast_to(zs.reshape(-1,1,1),self.nzyx)

    def __reset__(self):
        if not self.seed is None:
            torch.manual_seed(self.seed)
            np.random.seed(self.seed)
            self.current_idx=0

    def __len__(self):
        return self.length

    def __iter__(self):
        return self

    def __getitem__(self,idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if not type(idx) is list:
            if idx < self.current_idx:
                self.__reset__()
            self.__skip__(idx-self.current_idx)
            return self.__next__()
        else:
            res = []
            for i in idx:
                if i < self.current_idx:
                    self.__reset__()
                self.__skip__(i-self.current_idx)
                res.append(self.__next__())
        if len(res) == 1:
            return res[0]
        else:
            return res

    def __skip__(self,n):
        for i in range(n):
            self.current_idx += 1
            nlayers = np.random.choice(self.max_nlayers-1)+2 #从1到max_nlayers这个序列中随机采样，这里最大为7
            n_intrusion = np.random.choice(self.max_nintrusion) + 1 #确定密度体个数，最多为max_nintrusion个， 此处最大为2
            np.random.rand(nlayers) #
            np.random.rand(n_intrusion)
            np.random.rand(nlayers-1)
            for j in range(nlayers-1):
                np.random.rand(3)
            for j in range(n_intrusion):
                np.random.rand()
                np.random.rand()
                np.random.rand()
                np.random.rand()
                np.random.rand()
                np.random.rand() 

    @torch.no_grad()
    def __next__(self):
        '''
        Returns:
            field,density (tuple of Tensor): field with shape (1,ny,nx), density
                with shape (nz,ny,nx)
        '''
        self.current_idx += 1
        if self.current_idx > self.length:
            self.__reset__()
        density_model = torch.zeros(self.nzyx[0],self.nzyx[1],self.nzyx[2])
        r = density_model
        nlayers = np.random.choice(self.max_nlayers-1)+2 #从1到max_nlayers这个序列中随机采样，这里最大为7
        n_intrusion = np.random.choice(self.max_nintrusion) + 1 #确定密度体个数，最多为max_nintrusion个， 此处最大为2
        
        densities_layer = np.random.rand(nlayers)*(self.density_range[1]-self.density_range[0])+self.density_range[0] # 返回nlayers个密度值
        
        densities_layer = np.sort(densities_layer) #密度值排序
        densities_intrusion = np.random.rand(n_intrusion)*(self.anomly_density_range[1]-self.anomly_density_range[0])+self.anomly_density_range[0]
            #根据密度体个数生成异常密度值
            
        anchors = np.random.rand(nlayers-1)*self.dzyx[0]*self.nzyx[0] #锚定值
        
        density_model += densities_layer[0]
        
        for i in range(nlayers-1):
            normal = np.sin((np.random.rand(3)-0.5)*10*np.pi/180.)
            normal[2] = np.sqrt(1-normal[0]**2-normal[1]**2)
            density_model[np.dot((self.positions-np.array([0,0,anchors[i]])),normal)<0] = densities_layer[i+1]
            
        for i in range(n_intrusion):
            intrusion_center = np.array([(np.random.rand()-0.5)*self.dzyx[2]*self.nzyx[2],
                                         (np.random.rand()-0.5)*self.dzyx[1]*self.nzyx[1],
                                         np.random.rand()*self.dzyx[0]*self.nzyx[0]])
            intrusion_size = np.array([np.random.rand()*self.dzyx[2]*self.nzyx[2]*0.1,
                                       np.random.rand()*self.dzyx[1]*self.nzyx[1]*0.1,
                                       np.random.rand()*self.dzyx[0]*self.nzyx[0]*0.15])
            inside_box = (((self.positions-intrusion_center)<intrusion_size) &
                          ((self.positions-intrusion_center)>-1.*intrusion_size))
            inside_box = inside_box[:,:,:,0] & inside_box[:,:,:,1] & inside_box[:,:,:,2]
            density_model[inside_box] = densities_intrusion[i]
        gravity_field = self.decoder(density_model.unsqueeze(0))
        return gravity_field.squeeze(0), density_model

def load_data(args,seed=None,**kwargs):
    kwargs2 = {'num_workers': 1, 'pin_memory': False, 'drop_last': False}
    path = './models'
    data_length = args.n_batch * args.batch_size
    data_loader = torch.utils.data.DataLoader(
                DensityDataset(seed=seed,data_dir=path,length=data_length,**kwargs),
                batch_size=args.batch_size, shuffle=False, **kwargs2)
    return data_loader

def load_data_ddp(rank,args,seed=None):
    kwargs = {'num_workers': 1, 'pin_memory': False, 'drop_last': False}
    path = './models'
    data_length = args.n_batch * args.batch_size
    data_set = DensityDataset(seed=seed,data_dir=path,length=data_length)
#    print(len(data_set))
    data_sampler = torch.utils.data.distributed.DistributedSampler(
        data_set, 
        num_replicas=args.world_size,
        rank=rank
    )
    data_loader = torch.utils.data.DataLoader(
                dataset=data_set,
                batch_size=args.batch_size,
                shuffle=False,
                sampler=data_sampler,
                **kwargs)
    return data_loader