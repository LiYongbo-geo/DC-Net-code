import os
import sys
from datetime import datetime
import argparse
import numpy as np
import pathlib
from torch.utils.data import IterableDataset,Dataset
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch import optim
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter

import ops
import encoder
import decoder
import discriminator

def load_args():
    parser = argparse.ArgumentParser(description='gangravity')
    parser.add_argument('-c', '--checkpoint', default='./models/checkpointSH_withnoise_500m_adj2.pt',
                        type=str, help='checkpoint file')
    parser.add_argument('-d', '--device', default='cuda:0', type=str, help='computing device')
    parser.add_argument('-l', '--gp', default=10, type=int, help='gradient penalty')
    parser.add_argument('-g', '--n_gp', default=1, type=int)
    parser.add_argument('-b', '--batch_size', default=64, type=int)
    parser.add_argument('-n', '--n_batch', default=600, type=int)
    parser.add_argument('-e', '--epochs', default=600, type=int)
    parser.add_argument('-w', '--world_size', default=4, type=int)
    parser.add_argument('--use_spectral_norm', default=False)
    args = parser.parse_args()
    return args

def load_models(rank,args,checkpoint=None):
    netDec = decoder.GravDecoder()
    netDis = discriminator.GravDiscriminator()
    netEnc = encoder.GravEncoder()
    netEnc = DDP(netEnc.to(rank),device_ids=[rank])
    #netDec = DDP(netDec.to(rank),device_ids=[rank])
    netDec = netDec.to(rank)
    netDis = DDP(netDis.to(rank),device_ids=[rank])
    if checkpoint:
        netEnc.load_state_dict(checkpoint['enc_state_dict'])
        netDis.load_state_dict(checkpoint['dis_state_dict'])
#    netEnc = torch.nn.DataParallel(netEnc)
#    netDec = torch.nn.DataParallel(netDec)
#    netDis = torch.nn.DataParallel(netDis) 
    print (netDec, netDis, netEnc)
    return (netDec, netDis, netEnc)

def dice(pred, target):
    smooth = 1
    num = pred.size(0)
    m1 = pred.view(num, -1)  
    m2 = target.view(num, -1)  
    intersection = m1 * m2
    loss = (2. * intersection.sum(1) + smooth) / ((m1*m1).sum(1) + (m2*m2).sum(1) + smooth)
    return loss.sum()/num

def my_loss(pre_y, tru_y): 
    loss = 1 - dice(pre_y, tru_y)
    return loss

class DensityDataset(Dataset):
    def __init__(self, length):
        super(DensityDataset).__init__()
        self.current_idx = 0
        self.length = length

    def __reset__(self):
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
        modeltest = 'model{}.npy'.format(str(self.current_idx))
        testfile = os.path.join(os.getcwd(), 'traindatasetsSingleHeight_noise_adj', "models", modeltest)
        data = np.load(testfile)

        input_data = data[0, : , : ]
        input_data = input_data.reshape([1, 64, 64])
        oue_put = data[1, : , : ]
        return input_data, oue_put

def load_data(args):
    kwargs2 = {'num_workers': 1, 'pin_memory': False, 'drop_last': False}
    path = './models'
    data_length = args.n_batch * args.batch_size
    data_loader = torch.utils.data.DataLoader(
                DensityDataset(length=data_length),
                batch_size=args.batch_size, shuffle=False, **kwargs2)
    return data_loader

def load_data_ddp(rank,args):
    kwargs = {'num_workers': 1, 'pin_memory': False, 'drop_last': False}
    path = './models'
    data_length = args.n_batch * args.batch_size
    data_set = DensityDataset(length=data_length)
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

def train(rank,world_size,args):  
    # rank: 进程ID
    # world_size: 进程数
    setup(rank,world_size)
    #torch.manual_seed(1)
    if pathlib.Path(args.checkpoint).is_file():
        checkpoint = torch.load(args.checkpoint,map_location={'cuda:2':'cuda:{}'.format(rank)})
    else:
        checkpoint = None
    netDec, netDis, netEnc = load_models(rank,args,checkpoint)
    netDec.train()
    netDis.train()
    netEnc.train()
    #netG, netD, netE = load_models(args)

    if args.use_spectral_norm:
        optimizerDis = optim.Adam(filter(lambda p: p.requires_grad,
            netDis.parameters()), lr=2e-4, betas=(0.0,0.9))
    else:
        optimizerDis = optim.Adam(netDis.parameters(), lr=1e-3, betas=(0.9, 0.999))
    optimizerEnc = optim.Adam(netEnc.parameters(), lr=1e-3, betas=(0.9, 0.9),amsgrad=True)
    schedulerDis = optim.lr_scheduler.ExponentialLR(optimizerDis, gamma=0.99)
    schedulerEnc = optim.lr_scheduler.ExponentialLR(optimizerEnc, gamma=0.9)
    if checkpoint:
        optimizerDis.load_state_dict(checkpoint['dis_opt'])
        optimizerEnc.load_state_dict(checkpoint['enc_opt'])
        schedulerDis.load_state_dict(checkpoint['dis_sch'])
        schedulerEnc.load_state_dict(checkpoint['enc_sch'])
    optimizerEnc.param_groups[0]['lr'] = 1.0e-3
    optimizerEnc.param_groups[0]['betas'] = (0.9,0.99)
    one = torch.tensor(1.,device=args.device)
    mone = one * -1
    iteration = 0
    if checkpoint:
        iteration = checkpoint['iteration']+1
    writer = SummaryWriter('runs/experimentSH_withnoise_500m_adj3')
    start_time = datetime.now()
    tmp_time = start_time
    eps = 1.0e-8
    
    #开始训练
    for epoch in range(args.epochs):
        if rank == 2:
            print('enter epoch {}!!'.format(epoch))
        train_gen = load_data_ddp(rank,args)
        running_rec_loss = 0.
        running_gan_loss = 0.
        running_metrics= 0.
        check_sum = 0.
        data_iter = iter(train_gen)
        for i,(field_in, field_out) in enumerate(data_iter):
            field_in = field_in.type(torch.FloatTensor)
            field_out = field_out.type(torch.FloatTensor)
            """ Update Encoder """
            for p in netDis.parameters():
                p.requires_grad = False
            optimizerEnc.zero_grad()
            field_out = field_out.unsqueeze(1)
            field_out = field_out.to(rank)   #使用此命令使得运算在GPU上进行
            field_in = field_in.to(rank)

            field_rec = netEnc(field_in)
            #density_loss = F.mse_loss(density_rec,density)
            loss = F.mse_loss(field_rec, field_out)

            loss = loss.to(rank)
            metrics = torch.norm(field_rec - field_out)/(torch.norm(field_rec)+torch.norm(field_out))
            running_rec_loss += loss.item()
            running_metrics += metrics.item()

            loss.backward()
            optimizerEnc.step()
            if iteration == 0:
                #writer.add_graph(netDis,density)
                #writer.add_graph(netEnc,field)
                pass
            print_step = 1
            if iteration % print_step == (print_step-1):
                end_time = datetime.now()
                if rank == 0:
                    print(running_rec_loss)
                    print('epoch:{:<4d},Iteration:{:<9d},reconstruction_loss:{:<12.6f},metrics:{:<12.6f},time delta:{},totla time:{}'
                        .format(epoch,iteration,running_rec_loss/print_step,running_metrics/print_step,
                                end_time-tmp_time,
                                end_time-start_time))
                    writer.add_scalar('reconstruction_loss',running_rec_loss/print_step,iteration)
                    writer.add_scalar('metric',running_metrics/print_step,iteration)
                tmp_time = end_time
                running_rec_loss = 0.
                running_metrics= 0.
                running_gan_loss = 0.
                check_sum = 0.

            iteration += 1
            
        if rank == 0:
            netEnc.eval()
            train_length = int(args.n_batch * args.batch_size/10)
            tes_loss  = 0
            tes_metrix = 0
            for index_train in  range(train_length):
                index_train = index_train*10 +1
                with torch.no_grad():
                    gravtest = 'model{}.npy'.format(str(index_train))
                    gravfile = os.path.join(os.getcwd(), 'traindatasetsSingleHeight_noise_adj', "models", gravtest)
                    field = np.load(gravfile)
                    input_data = torch.from_numpy(field[0, :, :]).unsqueeze(0)
                    input_data = input_data.unsqueeze(0)
                    input_data = input_data.type(torch.FloatTensor) 
                    true_ = torch.from_numpy(field[1, :, :])
                    true_ = true_.type(torch.FloatTensor)
                    true_ = true_.to(rank)

                    density_rec = netEnc(input_data)
                    density_rec = density_rec.squeeze(0)
                    density_rec = density_rec.squeeze(0)

                    loss = F.mse_loss(density_rec, true_)
                    tes_loss += loss.item()
                    metrics = torch.norm(density_rec - true_)/(torch.norm(density_rec)+torch.norm(true_))
                    tes_metrix += metrics.item()

            writer.add_scalar('Train_loss',tes_loss/train_length,epoch)
            writer.add_scalar('Train_metric',tes_metrix/train_length,epoch)

        if rank == 0:
            netEnc.eval()
            verify_length = 200
            ver_sets = np.random.choice(1800, verify_length, replace = False) + 1
            ver_loss  = 0
            ver_metrix = 0
            for index_ver in ver_sets:
                with torch.no_grad():
                    
                    gravtest = 'model{}.npy'.format(str(index_ver))
                    gravfile = os.path.join(os.getcwd(), 'rdatasetsSingleHeight_noise_adj', "models", gravtest)
                    field = np.load(gravfile)
                    input_data = torch.from_numpy(field[0, :, :]).unsqueeze(0)
                    input_data = input_data.unsqueeze(0)
                    input_data = input_data.type(torch.FloatTensor) 
                    true_ = torch.from_numpy(field[1, :, :])
                    true_ = true_.type(torch.FloatTensor)
                    true_ = true_.to(rank)

                    density_rec = netEnc(input_data)
                    density_rec = density_rec.squeeze(0)
                    density_rec = density_rec.squeeze(0)
                    loss = F.mse_loss(density_rec, true_)

                    ver_loss += loss.item()
                    metrics = torch.norm(density_rec - true_)/(torch.norm(density_rec)+torch.norm(true_))
                    ver_metrix += metrics.item()  
            writer.add_scalar('Verify_loss',ver_loss/verify_length,epoch)
            writer.add_scalar('Verify_metric',ver_metrix/verify_length,epoch)

        if rank == 0:
            torch.save({'iteration':iteration,
                        'dis_state_dict':netDis.state_dict(),
                        'enc_state_dict':netEnc.state_dict(),
                        'dis_opt':optimizerDis.state_dict(),
                        'enc_opt':optimizerEnc.state_dict(),
                        'dis_sch':schedulerDis.state_dict(),
                        'enc_sch':schedulerEnc.state_dict()},
                    args.checkpoint)
    cleanup()
    
def setup(rank,world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '8844'
    dist.init_process_group("nccl", rank=rank,world_size=world_size) #初始化进程
    
def cleanup():
    dist.destroy_process_group()
    
def main():
    args = load_args()
    mp.spawn(train,args=(args.world_size,args),nprocs=args.world_size,join=True)
    
if __name__ == '__main__':
    torch.autograd.set_detect_anomaly(True)
    main()