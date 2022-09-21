import numpy as np
import torch
import torch.autograd as autograd

def calc_gradient_penalty(args, model, real_data, gen_data):
    alpha = torch.rand(args.batch_size,1,device=args.device)
    alpha = alpha.view(args.batch_size,1,1,1)
    interpolates = alpha * real_data + ((1 - alpha) * gen_data)
    interpolates = interpolates.clone().detach().requires_grad_(True)
    disc_interpolates = model(interpolates)
    gradients = autograd.grad(outputs=disc_interpolates,
                              inputs=interpolates,
                              grad_outputs=torch.ones_like(disc_interpolates,
                                                           device=args.device),
                              create_graph=True,
                              retain_graph=True, 
                              only_inputs=True)[0]
    gradient_penalty = ((gradients.reshape(gradients.shape[0],-1).norm(2, dim=1) - 1) ** 2).mean() * args.gp
    return gradient_penalty