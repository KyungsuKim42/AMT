import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import utils


data_path = '/home/data/kyungsu/AMT/processed/'
max_epoch = 500
max_patience = 20

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
    x = self.tanh1(self.pool1(self.conv1(x)))
    x = self.tanh2(self.pool2(self.conv2(x)))
    x = x.view(-1,7*30*50)
    x = self.sigm1(self.fc1(x))
    x = self.sigm2(self.fc2(x))
    x = self.fc3(x)
    
    return x

net = AMT()
def run_train(net,data,criterion,optimizer,batch_size=256):
  
  overall_loss = 0.0
  inputs, labels = data 
  inputs, labels = Variable(inputs), Variables(labels)
  
  num_samples = inputs.shape[0]
  num_batches = (num_samples+batch_size-1) / batch_size

  for i in range(num_batches):
    inputs_batch = inputs[i*batch_size:(i+1)*batch_size]
    labels_batch = labels[i*batch_size:(i+1)*batch_size]
    optimizer.zero_grad()
    outputs_batch = net(inputs_batch)
    loss = criterion(outputs_batch,labels_batch)
    loss.backward()
    optimizer.step()
    overall_loss = overall_loss + loss*inputs_batch.shape[0] 
  
  mean_loss = overall_loss / float(num_samples)
  
  return loss

def calculate_loss(net,data):


def main():
  net = AMT()
  train_x,train_y = data_load()
  test_x,test_y = data_load()
  
  train_x = utils.standardize(train_x,axis=1)
  test_y = utils.standardize(test_y,axis=1)

  train_x = utils.window(train_x,7) # shape = (N,7,264)
  test_x = utils.window(test_x,7) # shape = (M,7,264)
  
  min_valid_loss = float('inf') 
  patience = 0
  
  criterion = nn.BCELoss()
  optimizer = optim.SGD(net.parameters(),lr=0.001,momentum=0.9)
  for i in range(max_epoch):
    
    # Permutate train_x.
    train_x, trian_y = utils.permutate(train_x,train_y)
    
    # Train and calculate loss value.
    train_loss = run_train(net,(train_x,train_y),criterion,optimizer)
    valid_loss = run_loss(net,(test_x,test_y))
    
    if(valid_loss<min_valid_loss):
      patience = 0
      min_valid_loss = valid_loss
    else :
      patience += 1
    
    if(patience==20) break
    
    print '------{}th iteration max:{}-----'.format(i+1,max_epoch) 
    print 'train_loss : ' train_loss
    print 'valid_loss : ' valid_loss
    print 'patience : ' patience




