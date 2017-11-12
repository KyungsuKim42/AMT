from __future__ import print_function
import os

import pytorch_model
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import utils
import pdb


def main():

  model_file = '/home/kyungsukim/AMT/model/AMT_pytorch_model_ReLU_352'
  content_file = '/home/kyungsukim/AMT/wav/guitar_content.wav'
  style_file = '/home/kyungsukim/AMT/wav/guitar_style.wav'
  save_path = '/home/kyungsukim/AMT/result/cont=1.,1._style=1e-8,1e-8_optim=SGD_lr=0.1=_mom=0.9_/'
  content_coeffs = [1.,1.]
  style_coeffs = [1e-8,1e-8]
  num_iteration = 100000
  save_freq = 1000
  learning_rate = 0.1

  net = pytorch_model.AMT().cuda()
  net.load_state_dict(torch.load(model_file))
  for param in net.parameters():
      param.requires_grad = False
  print('model restored.')
  dtype = torch.cuda.FloatTensor

  content_wav= utils.load_wav(content_file)[:180000]
  style_wav = utils.load_wav(style_file)
  content_cqt = utils.standardize(utils.cqt(content_wav),axis=1)
  style_cqt = utils.standardize(utils.cqt(style_wav),axis=1)
  content_cqt = content_cqt.reshape([1,1,-1,264])
  style_cqt = style_cqt.reshape([1,1,-1,264])

  content_fname = 'content.npy'
  style_fname = 'style.npy'
  content_full_fname = os.path.join(save_path,content_fname)
  style_full_fname = os.path.join(save_path,style_fname)
  np.save(content_full_fname,content_cqt)
  np.save(style_full_fname,style_cqt)
  print('Content, Style CQT is saved.')


  content_var = Variable(torch.Tensor(content_cqt).type(dtype),requires_grad=False)
  style_var= Variable(torch.Tensor(style_cqt).type(dtype),requires_grad=False)

  stylized = stylize(net,content_var,style_var,content_coeffs,style_coeffs,
                     learning_rate,num_iteration,save_freq,save_path)
  np.save(save_path,stylized)


def stylize(net, content, style, content_coeffs, style_coeffs, learning_rate,
            num_iteration,save_freq,save_path):

  content_size = content.size()
  #combined = Variable(torch.from_numpy(np.random.normal(size = content_size,
    #                  scale = 0.1))).type(torch.cuda.FloatTensor)
  combined = Variable(torch.Tensor(np.random.normal(size=content_size,
                          scale=0.1)).cuda(),requires_grad=True)

  feats_content = net.features(content)
  grams_style = net.grams(style)

  loss_func = nn.MSELoss()
  optimizer = optim.SGD([combined],lr=learning_rate,momentum=0.9)
  for i in range(num_iteration):
    content_loss,style_loss = 0.,0.
    feats_combined = net.features(combined)
    grams_combined = net.grams(combined)

    feat_losses = []
    gram_losses = []
    for j in range(len(feats_combined)):
      feat_loss = content_coeffs[j]*loss_func(feats_combined[j],feats_content[j])
      feat_loss.backward(retain_graph=True)
      content_loss += feat_loss

    for j in range(len(grams_combined)):
      gram_loss = style_coeffs[j]*loss_func(grams_combined[j],grams_style[j])
      gram_loss.backward(retain_graph=True)
      style_loss += gram_loss

    print("{}th iteration\nContent Loss : {}\nStyle Loss : {}".format(i,
          content_loss.data.cpu().numpy(),style_loss.data.cpu().numpy()))
    optimizer.step()
    optimizer.zero_grad()

    if(i % save_freq == save_freq-1):
      stylized = combined.data.cpu().numpy()
      save_file_name = '{}th_combined-content:{:.3f}_style:{:.3f}.npy'.format(i+1,
        content_loss.data.cpu().numpy()[0],style_loss.data.cpu().numpy()[0])
      save_full_path = os.path.join(save_path,save_file_name)
      np.save(save_full_path,stylized)
      print('{}th model saved'.format(i+1))


  return combined.data.cpu().numpy()



if __name__ == '__main__':
    main()
