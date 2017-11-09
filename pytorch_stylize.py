import pytorch_model
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import numpy as np
import utils
import pdb


def main():

  model_path = '/home/kyungsukim/AMT/model/AMT_pytorch_model_ReLU_352'
  content_path = '/home/kyungsukim/AMT/wav/guitar_content.wav'
  style_path = '/home/kyungsukim/AMT/wav/guitar_style.wav'
  save_path = '/home/kyungsukim/AMT/result/guitar_combined_cqt.npy'
  content_coeffs = [1.,1.]
  style_coeffs = [1.,1.]
  num_iteration = 100
  learning_rate = 0.01

  net = pytorch_model.AMT().cuda()
  dtype = torch.cuda.FloatTensor

  content_wav= utils.load_wav(content_path)
  style_wav = utils.load_wav(style_path)
  content_cqt = utils.cqt(content_wav)
  style_cqt = utils.cqt(style_wav)
  content_cqt = content_cqt.reshape([1,1,-1,264])
  style_cqt = style_cqt.reshape([1,1,-1,264])


  content_var = Variable(torch.Tensor(content_cqt).type(dtype))
  style_var= Variable(torch.Tensor(style_cqt).type(dtype))

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
  loss_func = mse_loss
  optimizer = optim.SGD([combined],lr=learning_rate)

  for i in range(num_iteration):
    feats_combined = net.features(combined)
    grams_combined = net.grams(combined)
    for j in range(len(feats_combined)):
      feat_loss = content_coeffs[j]*loss_func(feats_combined[j],feats_content[j])
      loss = loss + feat_loss

    for j in range(len(grams_combined)):
      gram_loss = content_coeffs[j]*loss_func(grams_combined[j],grams_style[j])
      loss = loss + gram_loss

    optimizer.zero_grad()
    loss.backward(retain_graph=True)
    optimizer.step()
    print("{}th iteration\nLoss : {}".format(i,loss.data.cpu().numpy()))

  return combined.numpy()

def mse_loss(input,target):
    return torch.sum((input - target)*(input - target)) / input.data.nelement()

if __name__ == '__main__':
    main()
