import pytorch_model
import torch
from torch.autograd import Variable
import numpy as np
import pdb


def main():

  model_path = 'home/kyungsukim/AMT/model/AMT_pytorch_model_ReLU_352'
  num_iteration


pdb.set_trace()
dtype = torch.cuda.FloatTensor
data = np.ones([10,1,100,264])
input_var = Variable(torch.Tensor(data).type(dtype))

net = pytorch_model.AMT(7,264).cuda()

out = net.convolution(input_var)
