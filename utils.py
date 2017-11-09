import numpy as np
import os
import torch
import scipy.io.wavfile
import librosa

def permutate(x,y):
  """
  Permutate x and y array in same way.
  Args
    x,y (numpy array) : x and y data. shape = (num_samples,?,?)
  Returns
    x,y (numpy array) : Random permutated x,y data. shape = (num_samples,?,?)
  """
  dtype = torch.cuda.LongTensor
  perm = torch.randperm(x.size()[0]).type(dtype)
  return x[perm],y[perm]

def standardize(data,axis):
  """
  Standardize given data.
  data = (data-data.mean())/data.std()
  Args
    data (numpy array) : Input data. Assumed to be rank 2. shape = (?,?)
    axis (int) : Indicator of which axis to apply standardization.
  Returns
    data (numpy array) : Standardized data.
  """
  assert axis<22
  shape = list(data.shape)
  shape.pop(axis)
  for i in range(shape[0]):
      if (axis==0): data[:,i] = (data[:,i] - data[:,i].mean()) / data[:,i].std()
      elif (axis==1): data[i,:] = (data[i,:] - data[i,:].mean()) / data[i,:].std()

  return data

def window(data,window_size):
  """
  Slice data in window_wise.
  Args
    data (numpy array) : Input data. shape = (time,num_features)
    window_size (int) : Size of window.
  Returns
    windowed_data (numpy array) : Windowed data. shape=(time,window_size,num_features)
  """

  time_length = data.shape[0]
  num_features = data.shape[1]
  is_even = window_size % 2 == 0

  # Zero padding
  pad = np.zeros([window_size/2,num_features])
  pad_minus_one = np.zeros([window_size/2-1,num_features])
  data = np.append(pad,data,axis=0)
  if is_even:
    data = np.append(data,pad_minus_one,axis=0)
  else :
    data = np.append(data,pad,axis=0)

  # Append to list
  windowed_list = []
  for i in range(time_length):
    windowed_list.append(data[i:i+window_size])

  # Merge into single numpy array.

  windowed_data = np.asarray(windowed_list)

  return windowed_data

def data_load(path,max_files=0):
  """
  Load cqt and label data and return it as list.
  Order of cqt and label in each list is guaranteed to be same by using sort().
  Args
    path (python str) : Path from which we parse data.
  Returns
    x_list (list of np array) : list of numpy array of train data.
    y_list (list of np array) : list of numpy array of test data.
  """

  f_list = sorted(os.listdir(path))
  x_list = []
  y_list = []

  # Separate cqt and label file
  for i,f in enumerate(f_list):
    if (max_files!=0 and i>=max_files) : break
    if '.wav' in f:
      x_list.append(np.load(os.path.join(path,f)).T)
    elif '.txt' in f:
      y_list.append(np.load(os.path.join(path,f)).T)

  return x_list,y_list

def load_wav(path,target_sr=16000):
  """
  Load .wav file.
  Args
    path (python str) : Path from which we parse data.
  Returns
    sr (python float) : sample rate. default is 16000.
    wav_resample (numpy array) : resampled wav data. shape = [len].
  """
  original_sr,wav = scipy.io.wavfile.read(path)
  wav = 0.5*(wav[:,0]+wav[:,1])
  wav_resample = librosa.core.resample(wav,original_sr,target_sr)

  return wav_resample

def cqt(wav,sr=16000,hop_length=512,n_bins=264,bins_per_octave=36):
  """
  Calculate cqt
  Args
    wav (numpy array) : loaded wavfile. shape = [len]
  Returns
    cqt_wav : cqt result. shape = [?,?]
  """
  cqt_wav=np.abs(librosa.core.cqt(y=wav,sr=sr,
                 hop_length=hop_length,fmin=librosa.core.note_to_hz('A0'),
                 n_bins=n_bins,bins_per_octave=bins_per_octave))
  return cqt_wav
