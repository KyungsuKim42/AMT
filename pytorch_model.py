from __future__ import print_function

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
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
class AMT(nn.Module):
  
  def __init__(self):
    super(AMT,self).__init__()
    # Conv layers.
    self.conv1 = nn.Conv2d(1,50,(5,25),padding=(2,12))
    self.pool1 = nn.MaxPool2d((1,3),stride=(1,3))
    self.tanh1 = nn.Tanh() # TODO : Change into ReLU

    self.conv2 = nn.Conv2d(50,50,(3,5),padding=(1,2))
    self.pool2 = nn.MaxPool2d((1,3),padding=(0,1))
    self.tanh2 = nn.Tanh() # TODO : Change into ReLU

    # FC layers.
    self.fc1 = nn.Linear(7*30*50,1000)
    self.sigm1 = nn.Sigmoid() 
    self.fc2 = nn.Linear(1000,200)
    self.sigm2 = nn.Sigmoid() 
  
    # Output layer.
    self.fc3 = nn.Linear(200,88)
    self.sigm3 = nn.Sigmoid()
    
  def forward(self,x):
    x = x.view(-1,1,window_size,num_features)
    x = self.tanh1(self.pool1(self.conv1(x)))
    x = self.tanh2(self.pool2(self.conv2(x)))
    x = x.view(-1,7*30*50)
    x = self.sigm1(self.fc1(x))
    x = self.sigm2(self.fc2(x))
    x = self.fc3(x)
    
    return x


def run_train(net,inputs,labels,criterion,optimizer,batch_size=256):
  
  overall_loss = 0.0
  
  num_samples = inputs.size()[0]
  num_batches = (num_samples+batch_size-1) / batch_size

  for i in range(num_batches):
    input_batch = inputs[i*batch_size:(i+1)*batch_size]
    label_batch = labels[i*batch_size:(i+1)*batch_size]
    optimizer.zero_grad()
    output_batch = net(input_batch)
    loss = criterion(output_batch,label_batch)
    loss.backward()
    optimizer.step()
    overall_loss = overall_loss + loss*input_batch.size()[0] 
    print('progress : {}/{}'.format(i,num_batches),end='\r')

  mean_loss = overall_loss / float(num_samples)
  
  return loss


def run_loss(net,inputs,labels,criterion):


  outputs = net(inputs)
  loss = criterion(outputs,labels)

  return loss


def main():
  net = AMT().cuda()
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
    train_loss = run_train(net,train_x,train_y,criterion,optimizer,
                           batch_size).cpu().data.numpy()
    valid_loss = run_loss(net,test_x,test_y,criterion).cpu().data.numpy()
    if(valid_loss<min_valid_loss):
      patience = 0
      min_valid_loss = valid_loss
    else :
      patience += 1
    if(patience==20) : 
      torch.save(net.state_dict(),model_save_path+str(i+1))
      print('***{}th last model is saved.***'.format(i+1))
      break
    
    print ('------{}th iteration (max:{})-----'.format(i+1,max_epoch)) 
    print ('train_loss : ' ,train_loss)
    print ('valid_loss : ' ,valid_loss)
    print ('patience : ' ,patience)
    
    if ( i % save_freq == save_freq-1):
      torch.save(net.state_dict(),model_save_path+str(i+1))
      print ('***{}th model is saved.***'.format(i+1))
main()
