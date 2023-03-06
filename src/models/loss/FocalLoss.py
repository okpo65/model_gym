import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha, gamma, logits=False, reduce=True, device=torch.device('cpu')):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce
        self.device = device

    def forward(self, inputs, targets):

        ce_loss = torch.nn.functional.binary_cross_entropy(inputs, targets, reduction='none')
        targets = targets.view(-1).type(torch.LongTensor).to(self.device)

        pt = torch.exp(-ce_loss)
        F_loss = (1 - pt) ** self.gamma * ce_loss
        alpha = torch.tensor([self.alpha, 1 - self.alpha], dtype=torch.float).to(self.device)
        alpha_t = alpha.gather(0, targets)
        F_loss = alpha_t * F_loss
        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss

