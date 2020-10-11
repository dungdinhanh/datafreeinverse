import sys, os
import numpy as np
import copy
import torch
import torch.autograd as autograd
import torch.distributions as tdist

def setup_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)
    return path

def grid_mesh(X):
    # Plot the decision boundary
    # Determine grid range in x and y directions
    x_min, x_max = X[:, 0].min()-0.1, X[:, 0].max()+0.1
    y_min, y_max = X[:, 1].min()-0.1, X[:, 1].max()+0.1
    
    # Set grid spacing parameter
    spacing = min(x_max - x_min, y_max - y_min) / 100

    # Create grid
    XX, YY = np.meshgrid(np.arange(x_min, x_max, spacing), np.arange(y_min, y_max, spacing))

    # Concatenate data to match input
    data = np.hstack((XX.ravel().reshape(-1,1), YY.ravel().reshape(-1,1)))

    return XX, YY, data

def predict(XX, data, net):
    
    # Pass data to predict method
    db_prob = net(data)    
    
    clf = np.where(db_prob<0.5,0,1)
    
    Z = clf.reshape(XX.shape)
    Z_prob = db_prob.reshape(XX.shape)
    
    return db_prob, clf, Z, Z_prob

def calc_gradient_penalty(netD, real_data, fake_data, LAMBDA):
	alpha = torch.rand(real_data.shape[0], 1)
	alpha = alpha.expand(real_data.size())
	alpha = alpha.cuda()

	interpolates = alpha * real_data + ((1 - alpha) * fake_data)

	interpolates = interpolates.cuda()
	interpolates = autograd.Variable(interpolates, requires_grad=True)

	disc_interpolates = netD(interpolates)

	gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
							  grad_outputs=torch.ones(disc_interpolates.size()).cuda(),
							  create_graph=True, retain_graph=True, only_inputs=True)[0]

	gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
	return gradient_penalty
            
class bn_randomsample():
    
    def __init__(self, mean, covar):
        self.mean = mean
        self.covar = covar
        
        self.dist = tdist.Normal(mean, covar)
        #self.dist = tdist.MultivariateNormal(mean, covar)

    def samples(self, batch_size):
    
        return self.dist.sample((batch_size,))


def lr_policy(lr_fn):
    def _alr(optimizer, iteration, epoch):
        lr = lr_fn(iteration, epoch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    return _alr


def lr_cosine_policy(base_lr, warmup_length, epochs):
    def _lr_fn(iteration, epoch):
        if epoch < warmup_length:
            lr = base_lr * (epoch + 1) / warmup_length
        else:
            e = epoch - warmup_length
            es = epochs - warmup_length
            lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
        return lr

    return lr_policy(_lr_fn)


def add_noise_to_net(net, weight=0.01, noise_type='gaussian'):
	orig_params = []
	for params in net.parameters():
		orig_params.append(params.clone())
		if noise_type == 'gaussian':
		   noise = torch.randn_like(params)
		   params.data = params.data + weight * noise
		else:
		   noise = torch.rand_like(params)
		   params.data = params.data + weight * noise
	return net, orig_params

def transfer_weights(net, net_dis):
    for params, params_dis in zip(net.parameters(), net_dis.parameters()):
        params_dis = params.clone()
    return net_dis

def frozen_batchnorm(net):
    for module in net.modules():
        # print(module)
        if isinstance(module, nn.BatchNorm1d):
            if hasattr(module, 'weight'):
                module.weight.requires_grad_(False)
            if hasattr(module, 'bias'):
                module.bias.requires_grad_(False)
            module.eval()
    return net      

def reset_params(net, orig_params):
	for p, orig_p in zip(net.parameters(), orig_params):
	    p.data = orig_p.data
