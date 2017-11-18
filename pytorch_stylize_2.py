import pytorch_model
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import utils
import pdb


class ContentLoss(nn.Module):

  def __init__(self, target, weight):
    super(ContentLoss, self).__init__()
    # we 'detach' the target content from the tree used
    self.target = target.detach() * weight
    # to dynamically compute the gradient: this is a stated value,
    # not a variable. Otherwise the forward method of the criterion
    # will throw an error.
    self.weight = weight
    self.criterion = nn.MSELoss()

  def forward(self, input):
    self.loss = self.criterion(input * self.weight, self.target)
    self.output = input
    return self.output

  def backward(self, retain_graph=True):
    self.loss.backward(retain_graph=retain_graph)
    return self.loss

class GramMatrix(nn.Module):

  def forward(self, input):
    a, b, c, d = input.size()  # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)

    features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL

    G = torch.mm(features, features.t())  # compute the gram product

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(a * b * c * d)


class StyleLoss(nn.Module):

  def __init__(self, target, weight):
    super(StyleLoss, self).__init__()
    self.target = target.detach() * weight
    self.weight = weight
    self.gram = GramMatrix()
    self.criterion = nn.MSELoss()

  def forward(self, input):
    self.output = input.clone()
    self.G = self.gram(input)
    self.G.mul_(self.weight)
    self.loss = self.criterion(self.G, self.target)
    return self.output

  def backward(self, retain_graph=True):
    self.loss.backward(retain_graph=retain_graph)
    return self.loss



def main():

  model_path = '/home/kyungsukim/AMT/model/AMT_pytorch_model_ReLU_352'
  content_path = '/home/kyungsukim/AMT/wav/guitar_content.wav'
  style_path = '/home/kyungsukim/AMT/wav/guitar_style.wav'
  save_path = '/home/kyungsukim/AMT/result/guitar_combined_cqt.npy'
  content_coeffs = [0.0001,0.0001]
  style_coeffs = [0.0000001,0.0000001]
  num_iteration = 10
  learning_rate = 0.00000001

  net = pytorch_model.AMT().cuda()
  net.load_state_dict(torch.load(model_path))
  for param in net.parameters():
      param.requires_grad = False
  print('model restored.')
  dtype = torch.cuda.FloatTensor

  content_wav= utils.load_wav(content_path)[:180000]
  style_wav = utils.load_wav(style_path)
  content_cqt = utils.standardize(utils.cqt(content_wav),axis=1)
  style_cqt = utils.standardize(utils.cqt(style_wav),axis=1)
  content_cqt = content_cqt.reshape([1,1,-1,264])
  style_cqt = style_cqt.reshape([1,1,-1,264])

  content_var = Variable(torch.Tensor(content_cqt).type(dtype),requires_grad=False)
  style_var= Variable(torch.Tensor(style_cqt).type(dtype),requires_grad=False)

  stylized = stylize(net,content_var,style_var,content_coeffs,
                     style_coeffs,num_iteration,learning_rate)
  np.save(save_path,stylized)


def stylize(net, content, style, content_coeffs, style_coeffs, num_iteration,
            learning_rate):

  content_size = content.size()
  #combined = Variable(torch.from_numpy(np.random.normal(size = content_size,
    #                  scale = 0.1))).type(torch.cuda.FloatTensor)
  combined = Variable(torch.Tensor(np.random.normal(size=content_size,
                          scale=0.1)).cuda(),requires_grad=True)

  feats_content = net.features(content)
  grams_style = net.grams(style)

  loss = 0.
  loss_func = nn.MSELoss()
  optimizer = optim.SGD([combined],lr=learning_rate)
  for i in range(num_iteration):
    feats_combined = net.features(combined)
    grams_combined = net.grams(combined)

    for j in range(len(feats_combined)):
      feat_loss = content_coeffs[j]*loss_func(feats_combined[j],feats_content[j])
      loss = loss + feat_loss

    for j in range(len(grams_combined)):
      gram_loss = style_coeffs[j]*loss_func(grams_combined[j],grams_style[j])
      #loss = loss + gram_loss

    print("{}th iteration\nLoss : {}".format(i,loss.data.cpu().numpy()))

    loss.backward(retain_graph=True)
    optimizer.step()
    optimizer.zero_grad()

  return combined.data.cpu().numpy()



if __name__ == '__main__':
    main()
