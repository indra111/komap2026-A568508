"""Task-specific full-resolution residual refinement, related to shape streams.

Not a reproduction of Gated-SCNN and no claim of scientific novelty. The initial
function is exactly the saved baseline; four-class predictions remain the output.
"""
import torch
import torch.nn.functional as F
from pretrained_model import NormalizedUNet,build_optimizer as base_optimizer
from expansion import multiclass_loss

ARCHITECTURE='normalized_unet_resnet34_native_detail'


class NativeDetailNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.base=NormalizedUNet()
        self.refine=torch.nn.Sequential(
            torch.nn.Conv2d(7,24,3,padding=1),torch.nn.GroupNorm(6,24),torch.nn.SiLU(),
            torch.nn.Conv2d(24,24,3,padding=1),torch.nn.GroupNorm(6,24),torch.nn.SiLU(),
            torch.nn.Conv2d(24,4,1))
        torch.nn.init.zeros_(self.refine[-1].weight)
        torch.nn.init.zeros_(self.refine[-1].bias)

    @property
    def net(self):
        return self.base.net

    def forward(self,x):
        logits=self.base(x)
        # Native input bypasses encoder stride; base probabilities supply context.
        detail=torch.cat([2*x-1,logits.float().softmax(1)],1)
        return logits+self.refine(detail)


def build_optimizer(model):
    optimizer=base_optimizer(model)
    optimizer.add_param_group({'params':model.refine.parameters(),'lr':1e-3})
    return optimizer


def boundary_band(target):
    boundary=torch.zeros_like(target,dtype=torch.bool)
    dy=target[:,1:]!=target[:,:-1];boundary[:,1:]|=dy;boundary[:,:-1]|=dy
    dx=target[:,:,1:]!=target[:,:,:-1];boundary[:,:,1:]|=dx;boundary[:,:,:-1]|=dx
    return F.max_pool2d(boundary.float().unsqueeze(1),3,stride=1,padding=1)[:,0]>0


def training_loss(logits,target,method):
    base=multiclass_loss(logits,target)
    if method=='detail':return base
    if method!='detail_boundary':raise ValueError(method)
    ce=F.cross_entropy(logits.float(),target,reduction='none')
    band=boundary_band(target)
    values=[]
    for x,y in zip(ce,band):
        if y.any():values.append(x[y].mean())
    return base+.25*torch.stack(values).mean() if values else base


def load_detail(path,device):
    saved=torch.load(path,map_location='cpu',weights_only=True)
    if saved['architecture']!=ARCHITECTURE:raise ValueError('Unexpected detail architecture')
    model=NativeDetailNet();model.load_state_dict(saved['model'],strict=True)
    return model.to(device).eval(),saved
