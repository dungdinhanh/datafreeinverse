import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

from nets_mlp import netcls, netdis, netstd, netgen, netenc, weights_init
from hook     import bn1dfeathook, genbn1dfeathook
from utils    import add_noise_to_net, reset_params, calc_gradient_penalty
from lossfn   import knowledge_distill_loss, diveristy_loss

def dsm_score_estimation(net, loss_func, samples, y_gt, sigma=0.01):
    
    perturbed_samples = samples + torch.randn_like(samples) * sigma
    target = -1 / (sigma ** 2) * (perturbed_samples - samples)
    y_perturbed = net(perturbed_samples)
    #scores = torch.autograd.grad(loss_func(y_perturbed, y_gt), perturbed_samples, create_graph=True)[0]
    #target_loss = - 1 / 2. * ((y_perturbed - y_gt) ** 2).sum(dim=-1).mean(dim=0)
    target_loss = 1 / 2. * ((y_perturbed - y_gt) ** 2).mean()
    scores = torch.autograd.grad(target_loss, perturbed_samples, create_graph=True)[0]
    target = target.view(target.shape[0], -1)
    scores = scores.view(scores.shape[0], -1)
    #loss = 1 / 2. * ((scores - target) ** 2).sum(dim=-1).mean(dim=0)
    loss = 1 / 2. * ((scores - target) ** 2).mean()

    return loss

class network():
    
    def __init__(self, X, Y, n_hidden=4, lr=1e-2, n_samples = 128, basedir = 'outputs/',  device='cuda'):
        self.device = device
                
        self.X = X
        self.Y = Y.reshape(-1,1)
        self.Y_t = torch.FloatTensor(self.Y).to(device=self.device)
        print('data shape: ', self.X.shape)
        print('label shape: ', self.Y_t.shape)
        
        ''' hyper parameters '''
        self.n_input_dim   = X.shape[1]
        self.lr            = lr
        self.n_hidden      = n_hidden
        self.latent_dim    = 2
        self.label_dim     = 1
        self.n_output      = 1
        self.n_samples     = n_samples
        self.hidden_scale  = 70 #70

        self.classifier_lr = 0.025
        
        ''' classifier '''
        self.net = netcls(self.n_input_dim, self.n_hidden, self.n_output)
        self.net.apply(weights_init)
        
        ''' generator network, hidden_scale=70 (current best) '''
        self.net_gen = netgen(self.latent_dim, self.label_dim, self.n_input_dim, self.n_hidden, hidden_scale=self.hidden_scale)
        self.net_gen.apply(weights_init)
        
        ''' student network '''
        self.net_std = netdis(self.n_input_dim, int(self.n_hidden * 1 / 2), self.n_output)
        self.net_std.apply(weights_init)
        
        ''' student network '''
        self.net_enc = netenc(self.n_input_dim, self.latent_dim, self.n_hidden, self.latent_dim)
        self.net_enc.apply(weights_init)
                        
        if self.device == 'cuda':
            self.net.cuda()
            self.net_gen.cuda()
            self.net_std.cuda()
            self.net_enc.cuda()
        
        self.loss_func = nn.BCELoss()
        eps = 1e-9
        self.loss_func_continuous_label = lambda x, y : -((x+eps).log() * y + (1 - x + eps).log() * (1 - y)).mean()
        self.pretrain_optimizer = torch.optim.Adam(self.net.parameters(), lr=self.classifier_lr)
        
        ''' data '''
        self.basedir = basedir
        self.imgname = "plot_deepinversion_batch%d_classifierlr%0.3f_lr%0.3f_latentdim%d_hiddenscale%d" % (self.n_samples, self.classifier_lr, self.lr, self.latent_dim, self.hidden_scale)
        
    def predict(self, X):
        # Function to generate predictions based on data
        X_t = torch.FloatTensor(X).to(device=self.device)
        return self.net(X_t)
    
    def calculate_loss(self, y_hat):
        return self.loss_func(y_hat, self.Y_t)
    
    def update_network(self, y_hat):
        self.pretrain_optimizer.zero_grad()
        loss = self.calculate_loss(y_hat)
        loss.backward()
        self.pretrain_optimizer.step()
        self.training_loss.append(loss.item())
        
    def calculate_accuracy(self, y_hat_class, Y):
        return np.sum(Y.reshape(-1,1)==y_hat_class) / len(Y)
        
    def train(self, n_iters=1000, plotfig=True):
        print('Training...')
        self.training_loss = []
        self.training_accuracy = []
        
        if plotfig == True:
            fig, ax = plt.subplots(2, 1, figsize=(12,8))
            fig.show()
            fig.canvas.draw()
                    
            ax[0].set_ylabel('Loss')
            ax[0].set_title('Training Loss')

            ax[1].set_ylabel('Classification Accuracy')
            ax[1].set_title('Training Accuracy')

        for i in range(n_iters):
            y_hat = self.predict(self.X)
            self.update_network(y_hat)
            y_hat_class = np.where(y_hat.cpu()<0.5, 0, 1)
            accuracy = self.calculate_accuracy(y_hat_class, self.Y)
            self.training_accuracy.append(accuracy)
            print('Iter %d: %f' % (i, accuracy))
            
            if plotfig == True:
                if i % 50 == 0:
                   ax[0].plot(self.training_loss)
                   ax[1].plot(self.training_accuracy)
                
                   fig.canvas.draw() 
            
    def plot_training_results(self):
        fig, ax = plt.subplots(2, 1, figsize=(12,8))
        ax[0].plot(self.training_loss)
        ax[0].set_ylabel('Loss')
        ax[0].set_title('Training Loss')

        ax[1].plot(self.training_accuracy)
        ax[1].set_ylabel('Classification Accuracy')
        ax[1].set_title('Training Accuracy')

        plt.tight_layout()
        plt.show()
        
    def grid_mesh(self):
        # Plot the decision boundary
        # Determine grid range in x and y directions
        x_min, x_max = self.X[:, 0].min()-0.1, self.X[:, 0].max()+0.1
        y_min, y_max = self.X[:, 1].min()-0.1, self.X[:, 1].max()+0.1

        # Set grid spacing parameter
        spacing = min(x_max - x_min, y_max - y_min) / 100

        # Create grid
        XX, YY = np.meshgrid(np.arange(x_min, x_max, spacing),
                       np.arange(y_min, y_max, spacing))

        # Concatenate data to match input
        data = np.hstack((XX.ravel().reshape(-1,1), 
                          YY.ravel().reshape(-1,1)))


        ## Create hooks for feature statistics for batch norm
        loss_bn_feature_layers = []
               
        # Pass data to predict method
        db_prob = self.predict(data) 
        
        
        clf = np.where(db_prob.cpu()<0.5,0,1)
        
        Z = clf.reshape(XX.shape)
        Z_prob = db_prob.reshape(XX.shape)
                
        return XX, YY, data, db_prob, clf, Z, Z_prob
        
    def plot_testing_results(self, X_test, Y_test):
        # Pass test data
        y_hat_test = self.predict(X_test)
        y_hat_test_class = np.where(y_hat_test.cpu()<0.5, 0, 1)
        print("Test Accuracy {:.2f}%".format(
            self.calculate_accuracy(y_hat_test_class, Y_test) * 100))

        XX, YY, data, db_prob, clf, Z = self.grid_mesh()

        plt.figure(figsize=(12,8))
        plt.contourf(XX, YY, Z, cmap=plt.cm.Accent, alpha=0.5)
        plt.scatter(X_test[:,0], X_test[:,1], c=Y_test, 
                    cmap=plt.cm.Accent)
        plt.show()

    def setup_plot_progress(self, x = None):
        #fig, ax = plt.subplots(1, 3, figsize=(15, 5), gridspec_kw={'width_ratios': [1, 1]})
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        fig.show()
        fig.canvas.draw()
                
        # labels of figures
        ax[0].set_ylabel('Total loss')
        ax[0].set_ylim([0,10])
        ax[0].set_title('DeepInversion Loss')
        ax[1].set_title('Training and generated samples')
        ax[2].set_title('Classifier map')
        #ax[2].set_title('BN map')
                
        # plot the boundary
        XX, YY, data, db_prob, clf, Z, Z_prob = self.grid_mesh()
        ax[1].contourf(XX, YY, Z, cmap=plt.cm.Accent, alpha=0.5)  
        ax[1].scatter(self.X[:,0], self.X[:,1], c=self.Y, cmap=plt.cm.Accent)
        ax[2].scatter(XX, YY, c=Z_prob.cpu().detach().numpy(), cmap='viridis')
        #ax[2].scatter(XX, YY, c=bn_score.detach().numpy(), cmap='viridis')
        
        #fig = plt.figure(2)
        #ax = fig.add_subplot(111, projection='3d')
        #ax.plot_surface(XX, YY, Z_prob.detach().numpy())
        #ax.set_xlabel('X Label')
        #ax.set_ylabel('Y Label')
        #ax.set_zlabel('Z Label')
        
        #plt.show()
        
        #ax[2].scatter(self.X[:,0], self.X[:,1], c=self.Y, cmap=plt.cm.Accent)
        #ax[1].scatter(x_np[:,0], x_np[:,1], c='r', cmap=plt.cm.Accent)  
        #for i in range(np.shape(x_np)[0]): 
        #    ax[1].annotate('$P_%d$ (y=%d)' % (i, y_gt[i]), (x_np[i,0], x_np[i,1]))
        return fig, ax      

    def compute_loss_bn_gen(self,loss_bn_feature_layers_gen):
        self.bn_mean    = []
        self.bn_var     = []
        self.bn_weights = []
        self.bn_bias    = []

        bn_dis_cnt = 0
        for module in self.net.modules():
            if isinstance(module, nn.BatchNorm1d):
                self.bn_mean.append(module.running_mean.data)
                self.bn_var.append(module.running_var.data)
                self.bn_weights.append(module.weight)
                self.bn_bias.append(module.bias)
                bn_dis_cnt += 1
                print(module)
        print('Number of barch-norm layers (teacher network): ', bn_dis_cnt)
                
        bn_gen_cnt = 0
        for module in self.net_gen.modules():
            if bn_gen_cnt < bn_dis_cnt and isinstance(module, nn.BatchNorm1d):
               loss_bn_feature_layers_gen.append(genbn1dfeathook(module, self.bn_mean[bn_gen_cnt], self.bn_var[bn_gen_cnt], \
                                                                        self.bn_weights[bn_gen_cnt], self.bn_bias[bn_gen_cnt]))
               bn_gen_cnt += 1        
               print(module)
        print('Number of barch-norm layers (generator network): ', bn_gen_cnt + bn_dis_cnt)
        return loss_bn_feature_layers_gen
        
        
    def deepinversion(self, use_generator = False, discrete_label=True, knowledge_distill = 0.0, n_iters = 100):
        
        if use_generator == True:
            z    = torch.randn((self.n_samples, self.latent_dim), requires_grad=False, device=self.device, dtype=torch.float)
            if discrete_label == True:
               y_gt = torch.randint(0, 2, (self.n_samples, self.label_dim), dtype=torch.float, device=self.device)
            else:
               y_gt = torch.cuda.FloatTensor(self.n_samples, self.label_dim).uniform_(0, 1)
            x    = self.net_gen(z, y_gt)
            optimizer = torch.optim.Adam(self.net_gen.parameters(), lr=self.lr) #original 0.25
        else:
            x    = torch.randn((self.n_samples, 2), requires_grad=True, device=self.device, dtype=torch.float)
            y_gt = torch.randint(0, 2, (self.n_samples, 1), dtype=torch.float, device=self.device)
            optimizer = torch.optim.Adam([x], lr=self.lr) #original 0.25

        # plot the figure
        x_np = x.cpu().detach().clone().numpy()
        fig, ax = self.setup_plot_progress(x_np)
        
        # store the total loss
        total_loss = []
        
        # set for testing with batchnorm
        self.net.eval()
        
        ## Create hooks for feature statistics
        loss_bn_feature_layers = []

        for module in self.net.modules():
            if isinstance(module, nn.BatchNorm1d):
                loss_bn_feature_layers.append(bn1dfeathook(module))
                
        sigma_init = 0.01
        sigma_max  = 1.0
        sigma = np.arange(sigma_init, sigma_max, (sigma_max - sigma_init)/n_iters)
        #print(n_iters, len(sigma))
        #exit()
        
        #lr_scheduler = lr_cosine_policy(self.lr, 100, n_iters)
        
        for it in range(n_iters):
            self.net.zero_grad()
            self.net_gen.zero_grad()
            optimizer.zero_grad()
            
            #lr_scheduler(optimizer, it, it)
            
            #z = torch.randn((self.n_samples, self.latent_dim), requires_grad=False, device=self.device, dtype=torch.float)
            
            if use_generator == True:
                x = self.net_gen(z, y_gt)
                
            #if eps_noise > 0:
            #    x_eps = x + eps_noise*torch.randn_like(x, requires_grad=False)
            
            y_pd = self.net(x)

            # target loss
            loss_main = self.loss_func(y_pd, y_gt)
            
            # l2 loss
            loss_l2  =  torch.norm(x.view(-1, self.n_input_dim), dim=1).mean()
            
            # bn loss
            rescale = [1. for _ in range(len(loss_bn_feature_layers))]
            loss_bn = sum([mod.r_feature * rescale[idx] for (idx, mod) in enumerate(loss_bn_feature_layers)])
            
            #if use_generator == True:
            #   rescale_gen = [1. for _ in range(len(loss_bn_feature_layers_gen))]
            #   loss_bn_gen = sum([mod.r_feature * rescale_gen[idx] for (idx, mod) in enumerate(loss_bn_feature_layers_gen)])
            
            #loss_grad = dsm_score_estimation(self.net, self.loss_func, x, y_gt, sigma=sigma[-(it+1)])
            
            # loss
            #loss =  0.01 * loss_main + 0.005 * loss_l2 + 0.5 * loss_bn #+ 0.25 * loss_grad
            #loss =  loss_main + 0.005 * loss_l2 + 0.5 * loss_bn #+ 0.25 * loss_grad
            loss =  loss_main + 0.05 * loss_l2 + 0.5 * loss_bn #+ 0.25 * loss_grad
            
            #if use_generator == True:
            #   loss += 0.05 * loss_bn_gen
               
            total_loss.append(loss.item())
            
            loss.backward()
            optimizer.step()
            
            if it % 10 == 0:  
               print('-- iter %d --' % (it))
               print('target loss: %f' % (loss_main.item()))
               print('l2 loss: %f' % (loss_l2.item()))
               print('bn loss: %f' % (loss_bn.item()))
               #print('grad loss: %f' % (loss_grad.item()))
               print('loss: %f' % (loss.item()))
               ax[0].plot(total_loss, c='b')
               
               fig.canvas.draw()
       
        x_np = x.cpu().detach().numpy()  
       
        ax[1].scatter(x_np[:,0], x_np[:,1], c='b', cmap=plt.cm.Accent) 
        #for i in range(np.shape(x_np)[0]):    
        #    ax[1].annotate('$P_%d^*$ (y=%d)' % (i, y_gt[i]), (x_np[i,0], x_np[i,1]))
        plt.savefig(self.basedir + "fig_deepinversion_loss.png")
        plt.show()
        
        for name, param in self.net_gen.named_parameters():
            print(name)
            print(param)

    ''' improving the training of deepinversion '''
    def deepinversion_improved(self, use_generator      = False, \
                                     discrete_label     = True,  \
                                     noisify_network    = 0.0, \
                                     knowledge_distill  = 0.0, \
                                     mutual_info        = 0.0, \
                                     batchnorm_transfer = 0.0, \
                                     use_discriminator  = 0.0, \
                                     n_iters = 100):
        tb = SummaryWriter()
        if use_generator == True:
            z    = torch.randn((self.n_samples, self.latent_dim), requires_grad=False, device=self.device, dtype=torch.float)
            if discrete_label == True:
               y_gt = torch.randint(0, 2, (self.n_samples, self.label_dim), dtype=torch.float, device=self.device)
            else:
               y_gt = torch.cuda.FloatTensor(self.n_samples, self.label_dim).uniform_(0, 1)
            x    = self.net_gen(z, y_gt)
            if mutual_info > 0.0:
                ''' declare the optimizer for the encoder network '''
                optimizer = torch.optim.Adam(list(self.net_gen.parameters()) + list(self.net_enc.parameters()), lr=self.lr)
            else:
                optimizer = torch.optim.Adam(self.net_gen.parameters(), lr=self.lr)
        else:
            x    = torch.randn((self.n_samples, 2), requires_grad=True, device=self.device, dtype=torch.float)
            if discrete_label == True:
               y_gt = torch.randint(0, 2, (self.n_samples, self.label_dim), dtype=torch.float, device=self.device)
            else:
               y_gt = torch.cuda.FloatTensor(self.n_samples, self.label_dim).uniform_(0, 1)
            optimizer = torch.optim.Adam([x], lr=self.lr)
            
        #update name of output
        self.imgname = self.imgname + "_gen%d" % (use_generator)

        ''' declare the optimizer for the student network '''
        optimizer_std = torch.optim.Adam(self.net_std.parameters(), lr=self.classifier_lr)

        if self.device == 'cuda':
           x_np = x.cpu().detach().clone().numpy()
        else:
           x_np = x.detach().clone().numpy()
        
        fig, ax = self.setup_plot_progress(x_np)
        
        total_loss = []
        
        # set for testing with batchnorm
        self.net.eval()
        
        ## Create hooks for feature statistics
        loss_bn_feature_layers = []
        if use_generator == True and use_discriminator > 0.0:
           nets_dis = []
           nets_dis_params = []

        for module in self.net.modules():
            if isinstance(module, nn.BatchNorm1d):
                loss_bn_feature_layers.append(bn1dfeathook(module))
                if use_generator == True and use_discriminator > 0.0:
                    net_dis = netdis(module.running_mean.shape[0], self.n_hidden, 1).cuda()
                    net_dis.apply(weights_init)
                    nets_dis.append(net_dis)
                    nets_dis_params += list(net_dis.parameters())

        if use_generator == True and use_discriminator > 0.0:
           self.optimizer_dis = torch.optim.Adam(nets_dis_params, lr=self.lr, betas=(0.5, 0.9))
        
        ## Create hooks for feature statistics for generator        
        if use_generator == True and batchnorm_transfer> 0.0:
           loss_bn_feature_layers_gen = []
           self.compute_loss_bn_gen(loss_bn_feature_layers_gen)
        
        for it in range(n_iters):
            self.net.zero_grad()
            self.net_gen.zero_grad()
            self.net_std.zero_grad()
            self.net_enc.zero_grad()
            optimizer.zero_grad()
            optimizer_std.zero_grad()
            
            if use_generator == True:
               ''' randomly sampling latent and labels '''  
               z = torch.randn((self.n_samples, self.latent_dim), requires_grad=False, device=self.device, dtype=torch.float)
               y_gt = torch.randint(0, 2, (self.n_samples, self.label_dim), dtype=torch.float, device=self.device)
               
            if use_generator == True:
                ''' generating samples with generator '''
                x = self.net_gen(z, y_gt)
            
            '''
            **********************************************************************
            To optimize the generated samples or training the generator
            **********************************************************************
            '''
            if noisify_network > 0.0:
                ''' adding noise into the pre-trained classifier '''
                weight = noisify_network * (n_iters - it) / n_iters
                self.net, orig_params = add_noise_to_net(self.net, weight=weight, noise_type='uniform')
                
            if it == 0:
               self.imgname = self.imgname + "_nosify%0.3f" % (noisify_network)
                        
            y_pd = self.net(x)

            ''' main loss (cross-entropy loss) '''
            loss_main = self.loss_func(y_pd, y_gt)
            
            ''' l2 regularization '''
            loss_l2  =  torch.norm(x.view(-1, self.n_input_dim), dim=1).mean()
                        
            ''' batch-norm regularization '''
            rescale = [1. for _ in range(len(loss_bn_feature_layers))]
            loss_bn = sum([mod.r_feature * rescale[idx] for (idx, mod) in enumerate(loss_bn_feature_layers)])

            ''' total loss '''
            if use_generator == True and use_discriminator > 0.0:
               bn_w = 0.05
            else:
               bn_w = 1.0
            
            loss =  loss_main + 0.005 * loss_l2 + bn_w * loss_bn
           
            if knowledge_distill > 0.0:
               ''' knowledge distillation (teacher-student) based regularization '''
               y_st = self.net_std(x)
               #loss_kd = 1 - self.loss_func(y_st, y_pd.detach())
               loss_kd = knowledge_distill_loss(y_pd.detach(), y_st)
               loss = loss + knowledge_distill * loss_kd
               
            if it == 0:
               self.imgname = self.imgname + "_kdistill%0.3f" % (knowledge_distill)
               
            if mutual_info > 0.0:
               ''' mutual information constraint '''
               ze = self.net_enc(x)
               loss_mi  = ((z - ze)**2).mean()
               
               zdiv = torch.randn((self.n_samples, self.latent_dim), requires_grad=False, device=self.device, dtype=torch.float)
               xdiv = self.net_gen(zdiv, y_gt)
               loss_div = diveristy_loss(z, x, zdiv, xdiv)
               
               loss = loss + mutual_info*loss_mi + 0.1*mutual_info*loss_div
               
            if it == 0:
               self.imgname = self.imgname + "_minfo%0.3f" % (mutual_info)
               
            if use_generator == True and batchnorm_transfer > 0.0:
               ''' batch-norm transfer loss '''
               rescale_gen = [1. for _ in range(len(loss_bn_feature_layers_gen))]
               loss_bn_gen = sum([mod.r_feature * rescale_gen[idx] for (idx, mod) in enumerate(loss_bn_feature_layers_gen)])
               loss = loss + batchnorm_transfer * loss_bn_gen

            if it == 0:
               self.imgname = self.imgname + "_btransfer%0.3f" % (batchnorm_transfer)

            if use_generator == True and use_discriminator > 0.0:
                # train the generator on features
                loss_g = 0
                # traing the generator on features
                for (idx, mod) in enumerate(loss_bn_feature_layers):
                   nets_dis[idx].zero_grad()
                   # frozen the gradient for the discriminator
                   for p in nets_dis[idx].parameters():
                       p.requires_grad = False  # to avoid computation
                   feat_fake   = mod.feat_fake.cuda()
                   d_fake      = nets_dis[idx](feat_fake)
                   loss_g      = loss_g - d_fake.mean()
                loss = loss + use_discriminator * loss_g

            if use_generator == True and it == 0:
               self.imgname = self.imgname + "_discriminator%0.3f" % (use_discriminator)
                                          
            loss.backward()
            optimizer.step()

            if it % 100 == 0:
                tb.add_scalar("Total loss: ", loss, it)
                tb.add_scalar("Loss batchnorm", loss_bn, it)
                tb.add_histogram("Input", x, it)
                # tb.add_histogram("Input/gradients", x.grad, it)
                net_gen_state_dict = self.net_gen.state_dict()
                for key, value in net_gen_state_dict.items():
                    tb.add_histogram(key, value)
                    tb.add_histogram(key + "/gradient", value.grad, it)

            if noisify_network > 0.0:
               ''' reset the network's parameters '''
               reset_params(self.net, orig_params)

            if knowledge_distill > 0.0:
               '''
               **********************************************************************
               To update the student network
               **********************************************************************
               '''
               if use_generator == True:
                  ''' generating samples with generator '''
                  x = self.net_gen(z, y_gt)
               
               y_pd = self.net(x)
               y_st = self.net_std(x)
               #loss_kd = self.loss_func(y_st, y_pd.detach())
               loss_kd = 1. - knowledge_distill_loss(y_pd.detach(), y_st)
               loss_kd.backward()
               optimizer_std.step()
            
            ''' store the main loss to plot on the figure '''
            total_loss.append(loss.item())

            if use_generator == True and use_discriminator > 0.0:
                # traing the discriminator on features
                for _ in range(5):
                    loss_d = 0
                    x = self.net_gen(z, y_gt)
                    self.net(x)
                    for (idx, mod) in enumerate(loss_bn_feature_layers):
                        nets_dis[idx].zero_grad()
                        for p in nets_dis[idx].parameters():  # reset requires_grad
                            p.requires_grad = True 
                        feat_real = mod.feat_real.cuda()
                        feat_fake = mod.feat_fake.cuda()
                        d_real    = nets_dis[idx](feat_real)
                        d_fake    = nets_dis[idx](feat_fake)
                        penalty   = calc_gradient_penalty(nets_dis[idx], feat_real, feat_fake, LAMBDA=1.0)                 
                        loss_d    = loss_d + use_discriminator * (d_fake.mean() - d_real.mean() + penalty)
                    loss_d.backward()
                    self.optimizer_dis.step()

            if it % 10 == 0:  
               print('-- iter %d --'   % (it))
               print('target loss: %f' % (loss_main.item()))
               print('l2-norm loss: %f'     % (loss_l2.item()))
               print('batchnorm loss: %f'     % (loss_bn.item()))
               if knowledge_distill > 0.0:
                  print('distillation loss: %f'     % (loss_bn.item()))
               if mutual_info > 0.0:
                  print('mutual information / diversity losses: %f / %f' % (loss_mi.item(), loss_div.item()))
               if batchnorm_transfer > 0.0:
                  print('batch-norm transfer loss: %f ' % (loss_bn_gen.item()))
               if use_generator == True and use_discriminator > 0.0:
                  print('loss d / loss g: %f / %f' % (loss_d.item(), loss_g.item()))
               print('total loss: %f'  % (loss.item()))

               ''' realtime plot '''
               ax[0].plot(total_loss, c='b')
               fig.canvas.draw()
                       
        if self.device == 'cuda':
           x_np = x.cpu().detach().numpy()
        else:
           x_np = x.detach().numpy()
        tb.close()
        ax[1].scatter(x_np[:,0], x_np[:,1], c='b', cmap=plt.cm.Accent)
        plt.savefig(self.basedir + "%s.png" % (self.imgname))
        plt.show()

