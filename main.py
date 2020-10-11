import numpy as np
from   sklearn.datasets import make_moons
from   sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from   utils import setup_dir

#%matplotlib inline
print("Using PyTorch Version %s" %torch.__version__)

from network import network

basedir = setup_dir('outputs/')


''' Training data generation '''
np.random.seed(0)
torch.manual_seed(0)
X, Y = make_moons(500, noise=0.05)

# Split into test and training data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=73)

'''
plt.figure(figsize=(12,8))
plt.scatter(X_train[:,0], X_train[:,1], c=Y_train)
plt.title('Moon Data')
plt.show()
'''

''' algorithm's options '''
baseline           = False #use baseline deep inversion.
discrete_label     = True  #use discrete labels or continous labels [true is best].
use_generator      = True  #use the generator with deepinversion model.
knowledge_distill  = 0.00  #transfer knowledge to student network (default 0.02).
noisify_network    = 0.00  #add noise to the pre-trained classifier, this value is the weight of noise. (default 0.1).
mutual_info        = 0.00  #reconstruct the latent samples to be idential to the original latent inputs (default 0.1).
batchnorm_transfer = 0.00  #transfer batchnorm from classifier to generator (default 0.02).
use_discriminator  = 0.00  #adversarial training with the discriminator based on batch-norm (default 0.01).

''' hyper-parameters '''
n_samples          = 128   #batch size
if use_generator == True:
	lr = 0.001 #* (n_samples / 128.)**0.5
else:
	lr = 0.025

net = network(X, Y, n_hidden=15, lr=lr, n_samples=n_samples, basedir=basedir)

''' training the network '''
net.train(n_iters=1000, plotfig=False)
#net.plot_training_results()

# plot decision boundary
#net.plot_testing_results(X_train, Y_train)
#net.plot_testing_results(X_test, Y_test)

'''
Optimized with deep inversion, important hyper-parameters and observations:
1. The learning rate of classifier.
2. The learning rate of optimizer for generator or samples.
3. The size of generator network (finding the correct size is important).
4. The batch-norm normalization.
5. Adding noise to the pre-trained network is not too helpful on 2d-toy dataset.
6. Knowledge distillation does not help much on 2d-toy dataset.
'''
if baseline == True:
   net.deepinversion(use_generator     = use_generator,     \
                     discrete_label    = discrete_label,    \
                     knowledge_distill = knowledge_distill, \
                     n_iters=1000)
else:
   net.deepinversion_improved(use_generator      = use_generator,     \
                              discrete_label     = discrete_label,    \
                              knowledge_distill  = knowledge_distill, \
                              noisify_network    = noisify_network,   \
                              mutual_info        = mutual_info,       \
                              batchnorm_transfer = batchnorm_transfer,\
                              use_discriminator  = use_discriminator, \
                              n_iters            = 1000)
