import torch
import torch.nn as nn

from utils  import bn_randomsample

class bn1dfeathook():
    '''
    Implementation of the forward hook to track feature statistics and compute a loss on them.
    Will compute mean and variance, and will use l2 as a loss
    '''
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        
        mean = input[0].mean([0])
        var  = input[0].var(0, unbiased=False)
        
        # hook co compute deepinversion's feature distribution regularization
        #nch  = input[0].shape[1]
        #mean = input[0].mean([0, 2, 3])
        #var  = input[0].permute(1, 0, 2, 3).contiguous().view([nch, -1]).var(1, unbiased=False)

        #forcing mean and variance to match between two distributions
        #other ways might work better, i.g. KL divergence
        r_feature = torch.norm(module.running_var.data - var, 2) + torch.norm(
            module.running_mean.data - mean, 2)

        self.r_feature = r_feature
        # must have no output

        # generating the features 
        sampler = bn_randomsample(module.running_mean.data, module.running_var.data)
        samples = sampler.samples(input[0].shape[0])

        # save real and features
        self.feat_fake = input[0]
        self.feat_real = samples

    def close(self):
        self.hook.remove()

class genbn1dfeathook():
    '''
    Implementation of the forward hook to track feature statistics and compute a loss on them.
    Will compute mean and variance, and will use l2 as a loss
    '''
    def __init__(self, module, mean, var, weight, bias):
        self.mean   = mean
        self.var    = var
        self.weight = weight
        self.bias   = bias
        self.hook   = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        # hook co compute deepinversion's feature distribution regularization
        
        # r1                
        r_feature = torch.norm(torch.matmul((self.mean - module.bias).t(), module.running_var.data) - torch.matmul(module.weight.t(), self.bias - module.running_mean.data),1)
        
        # r2
        #r_feature = torch.norm(torch.matmul((self.mean).t(), module.running_var.data) - torch.matmul(module.weight.t(), - module.running_mean.data),1)
        
        self.r_feature = r_feature
        # must have no output
        
    def close(self):
        self.hook.remove()
        
