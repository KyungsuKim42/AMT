import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pdb
import glob
import os

data_dir = os.path.join(os.path.dirname(__file__),'data')
x_data_file_list = sorted(glob.glob(os.path.join(data_dir,'*.wav.npy')))
y_data_file_list = sorted(glob.glob(os.path.join(data_dir,'*.txt.npy')))

x_data_npy_list = [np.load(f) for f in x_data_file_list]
y_data_npy_list = [np.load(f) for f in y_data_file_list]

pdb.set_trace()

def shape(tensor):
    s = tensor.get_shape()
    return tuple([s[i].value for i in range(0,len(s))])

class AMT():
    """The AMT model."""
    def conv2d(self,input,filter_shape):
        init_f = tf.truncated_normal(filter_shape,stddev=0.01)
        init_b = tf.truncated_normal([filter_shape[-1]],stddev=0.01)
        
        filter = tf.get_variable(initializer=init_f)
        bias = tf.get_variable(initializer=init_b)
        
        conv = tf.nn.conv2d(input,filter,[1,1,1,1],padding='SAME')
        pre_activation = tf.nn.bias_add(conv,bias)
        activation = tf.nn.tanh(pre_activation)
        
        return activation
    
    def fully_connected(self,input,num_neurons):
        init_w = tf.truncated_normal([shape(input)[-1],num_neurons],stddev=0.01)
        init_b = tf.truncated_normal([num_neurons],stddev=0.01)
        w = tf.get_variable(initializer=init_w)
        b = tf.get_variable(initializer=init_b)
        
        logit = tf.nn.bias_add(tf.matmul(input,w),b)

        return logit


    def inference(self):
        """calculate output (predicted value) and loss.

        Return
          output(Tensor) : Predicted probability for each note. shape = [batch_size,num_pitches]
          loss(Tensor) : loss value. shape = [] (scalar)
        """
        
        input = self.x
        
        features = [] # for Neural Style.
        for (filter,pool) in zip(self.conv_filter,self.poolwin):
            features.append(conv2d(input,filter))
            input = tf.nn.max_pool(features[-1],pool)
        
        flattend = tf.reshape(input,[self.batch_size,-1])
        
        for num_neurons in fc_layer:
            logit = fully_connected(input,num_neurons)    
            input = tf.sigmoid(logit)

        git = fully_connected(input,num_pitches)
        loss = tf.nn.sigmoid_cross_entropy_with_logits(logit,self.y)
        output = tf.sigmoid(logit)

        return output,loss
    
    def __init__(self,config,sess):
        self.sess = sess
        
        self.win_size = config.win_size
        self.conv_filter = config.convfilter
        self.pool_win = config.poolwin
        self.fc_layer = config.fc_layer
        self.batch_size = config.batch_size
        self.learning_rate = config.learning_rate

    def batch_train(self,feed_dict):
        """train the network for a single batch
        Arg
          feed_dict(dict) : {x:x_batch,y:y_batch}
            x_batch(numpy array) : input data. shape = [batch_size,win_size,num_bins]
            y_batch(numpy array) : desired output data. shape = [batch_size,num_pitches]
        
        Return
          loss_value(numpy array) : calculated loss value. shape = [] (scalar)
        """

class TrainConfig():
    """Train Config."""
    win_size = 7
    batch_size = 256
    learning_rate = 0.001
    epoch = 10000
    num_pitches = 88
    num_bins = 264
    conv_filter = [(25,5,2,50),(5,3,50,50)] #(height(freq),width(time),input_ch,output_ch)
    pool_win = [(1,3,1,1),(1,3,1,1)] #(batch,height(freq),width(time),channels)
    fc_layer = [1000,200,num_pitches] # number of neurons of fully connected layer.
   


