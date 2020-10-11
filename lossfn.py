import torch
import torch.nn as nn

kl_loss = nn.KLDivLoss(reduction='batchmean').cuda()

def knowledge_distill_loss(outputs, outputs_student):
    T = 3.0
    P = nn.functional.softmax(outputs_student / T, dim=1)
    Q = nn.functional.softmax(outputs / T, dim=1)
    M = 0.5 * (P + Q)

    P = torch.clamp(P, 0.01, 0.99)
    Q = torch.clamp(Q, 0.01, 0.99)
    M = torch.clamp(M, 0.01, 0.99)
    eps  = 1e-5
    loss = 0.5 * kl_loss(torch.log(P + eps), M) + 0.5 * kl_loss(torch.log(Q + eps), M)
    # JS criteria - 0 means full correlation, 1 - means completely different
    loss = 1.0 - torch.clamp(loss, 0.0, 1.0)
    return loss
    
    
def diveristy_loss(z1, x1, z2, x2):
    return torch.norm(z1 - z2, p=1)/(torch.norm(x1 - x2, p=1)+1e-8).mean()
