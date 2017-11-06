import numpy as np

def permutate(x,y):
  """
  Permutate x and y array in same way.
  Args 
    x,y (numpy array) : x and y data. shape = (num_samples,?,?)
  Returns
    x,y (numpy array) : Random permutated x,y data. shape = (num_samples,?,?)
  """
  perm = np.random.permutation(x.shape[0])
  return x[perm],y[perm]

def standardize(data,axis):
  """
  Standardize given data.
  data = (data-data.mean())/data.std()
  Args
    data (numpy array) : Input data. Assumed to be rank 3. shape = (?,?,?)
    axis (int) : Indicator of which axis to apply standardization.
  Returns
    data (numpy array) : Standardized data.
  """
  assert axis<3
  shape = list(data.shape)
  shape.pop(axis)
  for i in range(shape[0]):
    for j in range(shape[1]):
      if (axis==0): data[:,i,j] = (data[:,i,j] - data[:,i,j].mean()) / data[:,i,j].std()
      elif (axis==1): data[i,:,j] = (data[i,:,j] - data[i,:,j].mean()) / data[i,:,j].std()
      else : data[i,j,:] = (data[i,j,:] - data[i,j,:].mean()) / data[i,j,:].std()

  return data

def window(data,window_size):
  """
  Slice data in window_wise.
  Args
    data (list of numpy array) : Input data. shape = [(time,num_features),...,]
    window_size (int) : Size of window.
  Returns
    windowed_data (numpy array) : Windowed data. shape=(batch*time,window_size,num_features)
  """
  
  num_features = data[0].shape[1]
  is_even = window_size % 2 == 0

  # Zero padding
  pad = np.zeros([window_size/2,num_features])
  pad_minus_one = np.zeros([window_size/2-1,num_features])
  for i,sample in enumerate(data):
    sample = np.append(pad,sample,axis=0)
    if is_even:
      sample = np.append(sample,pad_minus_one,axis=0)
    else :
      sample = np.append(sample,pad,axis=0)
    data[i] = sample 

  # Append to list
  windowed_list = []
  for sample in data:
    for i in range(sample.shape[0]-window_size+1):
      windowed_list.append(sample[i:i+window_size])

  # Merge into single numpy array.
  
  windowed_data = np.asarray(windowed_list)
  
  return windowed_data

