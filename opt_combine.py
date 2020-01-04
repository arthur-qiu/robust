import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class PGD_combine(nn.Module):
    def __init__(self, epsilon = 0.5, num_steps = 10, step_size = 0.1):
        super().__init__()
        self.epsilon = epsilon
        self.num_steps = num_steps
        self.step_size = step_size

    def forward(self, model, bx1, bx2, by = None):
        """
        :param model: the classifier's forward method
        :param bx: batch of images
        :param by: true labels
        :return: perturbed batch of images
        """

        alpha = torch.zeros_like(bx1) + 0.5
        x1 = bx1.detach().clone()
        x2 = bx2.detach().clone()
        bx_comb = alpha * x1 + (1.0-alpha) * x2
        if by is None:
            pre_logits = model(bx_comb * 2 - 1)
            if len(pre_logits) == 2:
                pre_logits = pre_logits[1]
            by = torch.argmax(pre_logits, 1)

        delta = torch.zeros_like(alpha).uniform_(-self.epsilon, self.epsilon)

        for i in range(self.num_steps):
            delta.requires_grad_()
            with torch.enable_grad():
                newx = (alpha+delta) * x1 + (1.0-alpha-delta) * x2
                logits = model(newx * 2 - 1)
                if len(logits) == 2:
                    logits = logits[1]
                loss = F.cross_entropy(logits, by, reduction='sum')
                loss = -loss # targeted

            grad = torch.autograd.grad(loss, delta, only_inputs=True)[0]

            delta = delta.detach() + self.step_size * torch.sign(grad.detach())

            delta = delta.clamp(-self.epsilon, self.epsilon)

        newx = (alpha + delta) * x1 + (1.0 - alpha - delta) * x2

        return newx

def flip(x, dim):
    indices = [slice(None)] * x.dim()
    indices[dim] = torch.arange(x.size(dim) - 1, -1, -1,
                                dtype=torch.long, device=x.device)
    return x[tuple(indices)]

class TenCrops(nn.Module):
    def __init__(self, pad = 4):
        super().__init__()
        self.pad = nn.ZeroPad2d(pad)

    def forward(self, x):
        pad_x = self.pad(x)
        x1 = flip(pad_x, 3)[:,:,0:32,0:32]
        x2 = flip(pad_x, 3)[:,:,0:32,8:40]
        x3 = flip(pad_x, 3)[:,:,8:40,8:40]
        x4 = flip(pad_x, 3)[:,:,8:40,0:32]
        x5 = flip(pad_x, 3)[:,:,4:36,4:36]
        x6 = pad_x[:, :, 0:32, 0:32]
        x7 = pad_x[:, :, 0:32, 8:40]
        x8 = pad_x[:, :, 8:40, 8:40]
        x9 = pad_x[:, :, 8:40, 0:32]
        x10 = pad_x[:, :, 4:36, 4:36]
        return(torch.cat([x1,x2,x3,x4,x5,x6,x7,x8,x9,x10],0))


class PGD_combine_10trs(nn.Module):
    def __init__(self, epsilon=0.5, num_steps=10, step_size=0.1):
        super().__init__()
        self.epsilon = epsilon
        self.num_steps = num_steps
        self.step_size = step_size
        self.TenCrops = TenCrops()

    def forward(self, model, bx1, bx2, by=None):
        """
        :param model: the classifier's forward method
        :param bx: batch of images
        :param by: true labels
        :return: perturbed batch of images
        """

        alpha = torch.zeros_like(bx1) + 0.5
        x1 = bx1.detach().clone()
        x2 = bx2.detach().clone()
        bx_comb = alpha * x1 + (1.0 - alpha) * x2
        if by is None:
            pre_logits = model(bx_comb * 2 - 1)
            if len(pre_logits) == 2:
                pre_logits = pre_logits[1]
            by = torch.argmax(pre_logits, 1)

        delta = torch.zeros_like(alpha).uniform_(-self.epsilon, self.epsilon)

        y = by.repeat(10)

        for i in range(self.num_steps):
            delta.requires_grad_()
            with torch.enable_grad():
                newx = (alpha + delta) * x1 + (1.0 - alpha - delta) * x2
                newx_ten = self.TenCrops(newx)
                logits = model(newx_ten * 2 - 1)
                if len(logits) == 2:
                    logits = logits[1]
                loss = F.cross_entropy(logits, y, reduction='sum')
                loss = -loss  # targeted

            grad = torch.autograd.grad(loss, delta, only_inputs=True)[0]

            delta = delta.detach() + self.step_size * torch.sign(grad.detach())

            delta = delta.clamp(-self.epsilon, self.epsilon)

        newx = (alpha + delta) * x1 + (1.0 - alpha - delta) * x2

        return newx



class CW_combine(nn.Module):
    pass