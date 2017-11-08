from __future__ import print_function
import torch
import torch.nn as nn

class AMT(nn.Module):

  def __init__(self,window_size,num_features):
    super(AMT,self).__init__()
    # Conv layers.
    self.window_size = window_size
    self.num_features = num_features
    self.conv1 = nn.Conv2d(1,50,(5,25),padding=(2,12))
    self.pool1 = nn.MaxPool2d((1,3),stride=(1,3))
    self.tanh1 = nn.ReLU() # TODO : Change into ReLU

    self.conv2 = nn.Conv2d(50,50,(3,5),padding=(1,2))
    self.pool2 = nn.MaxPool2d((1,3),padding=(0,1))
    self.tanh2 = nn.ReLU() # TODO : Change into ReLU

    # FC layers.
    self.fc1 = nn.Linear(7*30*50,1000)
    self.sigm1 = nn.Sigmoid()
    self.fc2 = nn.Linear(1000,200)
    self.sigm2 = nn.Sigmoid()

    # Output layer.
    self.fc3 = nn.Linear(200,88)
    self.sigm3 = nn.Sigmoid()

  def forward(self,x):
    x = x.view(-1,1,self.window_size,self.num_features)
    x = self.tanh1(self.pool1(self.conv1(x)))
    x = self.tanh2(self.pool2(self.conv2(x)))
    x = x.view(-1,7*30*50)
    x = self.sigm1(self.fc1(x))
    x = self.sigm2(self.fc2(x))
    x = self.fc3(x)
    return x

  def convolution(self,x):
    x = self.tanh1(self.pool1(self.conv1(x)))
    x = self.tanh2(self.pool2(self.conv2(x)))
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
