from datetime import datetime
import argparse
import numpy as np
import pathlib
import argparse

import torch

import datagen
import train
import utils

@torch.no_grad()
def evaluate(number=2,cut='x',**kwargs):
    args_gen = train.load_args()
    test_gen = datagen.load_data(args_gen)
    n_test = 4
    im_grid = (number//2,2)
    for i_test in range(n_test):
        field,density = iter(test_gen).next()
        field = field.to(args_gen.device)
        density = density.to(args_gen.device)
        print(density.cpu().numpy()[0])
        utils.plot_density(density.cpu().numpy()[0],
                           image_grid=im_grid,
                           cut=cut,
                           fname='./res/{:2d}_density_orig.png'.format(i_test))
        utils.plot_density(field.cpu().numpy()[0],
                           image_grid=im_grid,
                           cut='z',
                           fname='./res/{:2d}_field_orig.png'.format(i_test))
        total_field = torch.sum(field,axis=1)
        utils.plot_field(total_field.cpu().numpy()[0],fname='./res/{:2d}_total_field_orig.png'.format(i_test))

if __name__ == '__main__':
    evaluate(cut='z',number=2)
 
 