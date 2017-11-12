from __future__ import print_function

import numpy as np

import torch
import torch.nn.functional as F
from torch.autograd import Variable

iteration = 1000
lr = 0.001
x = Variable(torch.randn(5,2,2),requires_grad=True)
target = Variable(torch.randn(1,5,5))
optimizer = torch.optim.SGD([x],lr=lr)
loss = torch.nn.MSELoss()
#import pdb; pdb.set_trace()
for i in range(iteration):
  x_view = x.view(5,-1)
  gram = torch.mm(x_view,x_view.t())
  gram = gram / 4.
  gram = gram.view(1,5,5)
  gram = F.relu(F.max_pool2d(gram,(1,1)))
  output = loss(gram,target)
  output.backward(retain_graph=True)
  optimizer.step()
  optimizer.zero_grad()
  #x.data -= lr*x.grad.data
  #x.grad.data.zero_()
  print('loss : {}'.format(output))
'''

loss = torch.nn.MSELoss()
input = Variable(torch.randn(3,3),requires_grad=True)
m  = input + 3
target = Variable(torch.randn(3,3))
output = loss(m,target)
output.backward()
import pdb; pdb.set_trace()
'''
