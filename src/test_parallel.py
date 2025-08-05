import torch.nn as nn
import torch

m = nn.Linear(10, 10).to('cuda')
m = torch.nn.DataParallel(m)
x = torch.randn(8, 10).to('cuda')
print(m(x))
