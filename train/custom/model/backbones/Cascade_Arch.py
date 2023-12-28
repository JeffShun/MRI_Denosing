import torch
import torch.nn as nn

class Cascade_Arch(nn.Module):

    def __init__(self, base_model, num_iter, share_paras=True):
        super(Cascade_Arch, self).__init__()
        self.base_model = base_model
        self.share_paras = share_paras
        self.num_iter_total = num_iter

        if self.share_paras:
            self.num_iter = 1
        else:
            self.num_iter = num_iter

        self.models = torch.nn.ModuleList([base_model for i in range(self.num_iter)])

    def forward(self, x):          
        for k in range(self.num_iter_total):
            #dw 
            x = self.models[k%self.num_iter](x)
        return x
