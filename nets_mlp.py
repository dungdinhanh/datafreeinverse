import torch
import torch.nn as nn

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1: 
        m.running_var.data.normal_(1.0, 0.02)
        m.running_mean.data.fill_(0)
    #elif classname.find('CBN') != -1:
    #    m.bn.weight.data.normal_(1.0, 0.02)
    #    m.bn.bias.data.fill_(0)

'''
The network serves as the pre-trained classifier
'''
class netcls(nn.Module):
      def __init__(self, data_dim, n_hidden, n_output):
          super(netcls, self).__init__()
          self.data_dim   = data_dim
          self.n_hidden   = n_hidden
          self.n_output   = n_output
          self.out = nn.Sequential(
            nn.Linear(self.data_dim, self.n_hidden, bias=False),
            nn.BatchNorm1d(num_features=self.n_hidden),
            nn.ReLU(True),
            nn.Linear(self.n_hidden, self.n_output),
            nn.Sigmoid()
          )
      def forward(self, x):
          x = self.out(x)
          return x

'''
The network serves as the pre-trained classifier
'''
class netdis(nn.Module):
      def __init__(self, data_dim, n_hidden, n_output):
          super(netdis, self).__init__()
          self.data_dim   = data_dim
          self.n_hidden   = n_hidden
          self.n_output   = n_output
          self.out = nn.Sequential(
            #nn.Linear(self.data_dim, self.n_hidden, bias=False),
            #nn.BatchNorm1d(num_features=self.n_hidden),
            #nn.ReLU(True),
            #nn.Linear(self.n_hidden, self.n_output),
            nn.Linear(self.data_dim, self.n_output),
          )
      def forward(self, x):
          x = self.out(x)
          return x          
          
'''
Student network for knowledge distilation
'''
class netstd(nn.Module):
      def __init__(self, data_dim, n_hidden, n_output):
          super(netdis, self).__init__()
          self.data_dim   = data_dim
          self.n_hidden   = n_hidden
          self.n_output   = n_output
          self.out = nn.Sequential(
            nn.Linear(self.data_dim, self.n_hidden, bias=False),
            nn.BatchNorm1d(num_features=self.n_hidden),
            nn.ReLU(True),
            nn.Linear(self.n_hidden, self.n_output),
            nn.Sigmoid()
          )
      def forward(self, x):
          x = self.out(x)
          return x

class netgen(nn.Module):
      def __init__(self, latent_dim, label_dim, data_dim, n_hidden, hidden_scale=70):
          super(netgen, self).__init__()
          self.latent_dim = latent_dim
          self.data_dim   = data_dim
          self.g_hidden   = n_hidden
          self.label_dim  = label_dim
          
          '''
          # #current best: n_hidden * 60
          self.out = nn.Sequential (
            nn.Linear(self.latent_dim + self.label_dim, self.g_hidden, bias=False),
            nn.BatchNorm1d(num_features=self.g_hidden),
            nn.ReLU(True),
            nn.Linear(self.g_hidden, self.g_hidden, bias=False),
            nn.BatchNorm1d(num_features=self.g_hidden),
            nn.ReLU(True),
            nn.Linear(self.g_hidden, self.g_hidden, bias=False),
            nn.BatchNorm1d(num_features=self.g_hidden),
            nn.ReLU(True),
            nn.Linear(self.g_hidden, self.data_dim),
          )

          '''
          # our best with small batch-size
          # current best: n_hidden * 70
          '''
          g_hidden_big = self.g_hidden * hidden_scale
          self.out = nn.Sequential (
            nn.Linear(self.latent_dim + self.label_dim, g_hidden_big, bias=False),
            nn.BatchNorm1d(num_features=g_hidden_big),
            nn.ReLU(True),
            nn.Linear(g_hidden_big, self.data_dim)
          )
          '''
          
          '''
          # current best: n_hidden * 70
          g_hidden_big = self.g_hidden * hidden_scale
          self.out = nn.Sequential (
            nn.Linear(self.latent_dim + self.label_dim, self.g_hidden, bias=False),
            nn.BatchNorm1d(num_features=self.g_hidden),
            nn.ReLU(True),
            nn.Linear(self.g_hidden, g_hidden_big, bias=False),
            nn.BatchNorm1d(num_features=g_hidden_big),
            nn.ReLU(True),
            nn.Linear(g_hidden_big, self.data_dim)
          )
          '''
          # current best: n_hidden * 70
          g_hidden_big = self.g_hidden * hidden_scale
          self.out = nn.Sequential (
            nn.Linear(self.latent_dim + self.label_dim, g_hidden_big, bias=False),
            nn.BatchNorm1d(num_features=g_hidden_big),
            nn.ReLU(True),
            nn.Linear(g_hidden_big, g_hidden_big, bias=False),
            nn.BatchNorm1d(num_features=g_hidden_big),
            nn.ReLU(True),
            nn.Linear(g_hidden_big, self.data_dim)
          )

          '''
          #current best: n_hidden * 70, batch norm is important, it fails
          #if batch norm is removed.
          self.out = nn.Sequential (
            nn.Linear(self.latent_dim + self.label_dim, self.g_hidden, bias=False),
            nn.BatchNorm1d(num_features=self.g_hidden),
            nn.ReLU(True),
            nn.Linear(self.g_hidden, self.g_hidden * 70, bias=False),
            nn.BatchNorm1d(num_features=self.g_hidden * 70),
            nn.ReLU(True),
            nn.Linear(self.g_hidden * 70, self.data_dim)
          )
          '''
          '''
          #current best: n_hidden * 70, batch norm is important, it fails
          #if batch norm is removed.
          self.out = nn.Sequential (
            nn.Linear(self.latent_dim + self.label_dim, self.g_hidden, bias=False),
            nn.BatchNorm1d(num_features=self.g_hidden),
            nn.ReLU(True),
            nn.Linear(self.g_hidden, self.g_hidden * 35, bias=False),
            nn.BatchNorm1d(num_features=self.g_hidden * 35),
            nn.ReLU(True),
            nn.Linear(self.g_hidden * 35, self.g_hidden * 70, bias=False),
            nn.BatchNorm1d(num_features=self.g_hidden * 70),
            nn.ReLU(True),
            nn.Linear(self.g_hidden * 70, self.g_hidden * 35, bias=False),
            nn.BatchNorm1d(num_features=self.g_hidden * 35),
            nn.ReLU(True),
            nn.Linear(self.g_hidden * 35, self.data_dim)
          )
          '''
          
      def forward(self, z, y):
          z = torch.cat((z,y), dim=1)
          x = self.out(z)
          return x


class netenc(nn.Module):
      def __init__(self, data_dim, latent_dim, n_hidden, hidden_scale=70):
          super(netenc, self).__init__()
          self.latent_dim = latent_dim
          self.data_dim   = data_dim
          self.e_hidden   = n_hidden

          e_hidden_big = self.e_hidden * hidden_scale
          self.out = nn.Sequential (
            nn.Linear(self.data_dim, e_hidden_big, bias=False),
            nn.BatchNorm1d(num_features=e_hidden_big),
            nn.ReLU(True),
            nn.Linear(e_hidden_big, self.latent_dim)
          )
      def forward(self, x):
          z = self.out(x)
          return z
