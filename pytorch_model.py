from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F

class AMT(nn.Module):

  def __init__(self,window_size=7,num_features=264):
    super(AMT,self).__init__()

    # Model parameters.
    self.window_size = window_size
    self.num_features = num_features

    # Conv layers.
    self.conv1 = nn.Conv2d(1,50,(5,25),padding=(2,12))
    self.conv2 = nn.Conv2d(50,50,(3,5),padding=(1,2))

    # FC layers.
    self.fc1 = nn.Linear(7*30*50,1000)
    self.fc2 = nn.Linear(1000,200)

    # Output layer.
    self.fc3 = nn.Linear(200,88)

  def forward(self,x):
    x = x.view(-1,1,self.window_size,self.num_features)
    x = F.relu(F.max_pool2d(self.conv1(x),(1,3)))
    x = F.relu(F.max_pool2d(self.conv2(x),(1,3)))
    x = x.view(-1,7*30*50)
    x = F.sigmoid(self.fc1(x))
    x = F.sigmoid(self.fc2(x))
    x = self.fc3(x)
    return x

  # return feature map of each conv layer.
  def features(self,x):
    relu1 = F.relu(F.max_pool2d(self.conv1(x),(1,3)))
    relu2 = F.relu(F.max_pool2d(self.conv2(relu1),(1,3),padding=(0,1)))
    return relu1, relu2

  def grams(self,x):
    num_batches = x.size()[0]
    assert num_batches == 1 # Doesn't support batch grams yet.
    feature_list = self.features(x)
    g_list = []
    for feature in feature_list:
      a,b,c,d = feature.size()
      f = feature.view(b,c*d)
      g = torch.mm(f,f.t())
      g.div(b*c*d)
      g_list.append(g)

    return g_list


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
