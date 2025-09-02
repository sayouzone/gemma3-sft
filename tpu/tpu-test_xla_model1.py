# test_xla_model1.py
import torch
#import torch_xla.core.xla_model as xm
import torch_xla

#dev = xm.xla_device()
dev = torch_xla.device()
t1 = torch.randn(3,3,device=dev)
t2 = torch.randn(3,3,device=dev)
print(t1 + t2)