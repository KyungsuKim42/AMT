from __future__ import print_function

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_model
import numpy as np
import os.path
import utils
import pdb

data_path = '/home/data/kyungsu/AMT/processed/'
model_save_path = '/home/kyungsukim/AMT/model/AMT_pytorch_model'
save_freq = 100
max_epoch = 5000
max_patience = 20
window_size = 7
num_features = 264
batch_size = 256


def main():
  net = pytorch_model.AMT(window_size,num_features).cuda()
  train_x_list,train_y_list = utils.data_load(os.path.join(data_path,'train/'),2)
  test_x_list,test_y_list = utils.data_load(os.path.join(data_path,'test/'),2)

  # Standardize.
  for i in range(len(train_x_list)):
    train_x_list[i] = utils.standardize(train_x_list[i],axis=0)
    train_x_list[i] = utils.window(train_x_list[i],window_size)

  # Slice window with stride=1, pad='SAME'
  for i in range(len(test_x_list)):
    test_x_list[i] = utils.standardize(test_x_list[i],axis=0)
    test_x_list[i] = utils.window(test_x_list[i],window_size)

  train_x = np.vstack(train_x_list)
  train_y = np.vstack(train_y_list)
  test_x = np.vstack(test_x_list)
  test_y = np.vstack(test_y_list)

  # For GPU computing.
  dtype = torch.cuda.FloatTensor
  train_x = Variable(torch.Tensor(train_x).type(dtype))
  train_y = Variable(torch.Tensor(train_y).type(dtype))
  test_x = Variable(torch.Tensor(test_x).type(dtype))
  test_y = Variable(torch.Tensor(test_y).type(dtype))

  min_valid_loss = float('inf')
  patience = 0

  criterion = nn.BCEWithLogitsLoss()
  optimizer = optim.SGD(net.parameters(),lr=0.001,momentum=0.9)

  print ('Preprocessing Completed.')

  '''
  net.load_state_dict(torch.load(model_save_path+'10'))
  out = net(train_x[:10])
  print(out)
  '''
  for i in range(max_epoch):

    # Permutate train_x.
    train_x, trian_y = utils.permutate(train_x,train_y)

    # Train and calculate loss value.
    train_loss = pytorch_model.run_train(net,train_x,train_y,criterion,optimizer,
                           batch_size).cpu().data.numpy()
    valid_loss = pytorch_model.run_loss(net,test_x,test_y,criterion).cpu().data.numpy()
    if(valid_loss<min_valid_loss):
      patience = 0
      min_valid_loss = valid_loss
    else :
      patience += 1
    if(patience==20) :
      torch.save(net.state_dict(),model_save_path+'_ReLU_'+str(i+1))
      print('***{}th last model is saved.***'.format(i+1))
      break

    print ('------{}th iteration (max:{})-----'.format(i+1,max_epoch))
    print ('train_loss : ' ,train_loss)
    print ('valid_loss : ' ,valid_loss)
    print ('patience : ' ,patience)

    if ( i % save_freq == save_freq-1):
      torch.save(net.state_dict(),model_save_path+'_ReLU_'+str(i+1))
      print ('***{}th model is saved.***'.format(i+1))

if __name__ == '__main__':
    main()
