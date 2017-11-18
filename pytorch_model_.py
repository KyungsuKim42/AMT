import librosa
import numpy as np
import scipy.io.wavfile
import os.path
import glob
import pdb
import matplotlib.pyplot as plt
#Parameters of CQT
frame_per_sec = target_sr/float(hop_length) # 31.25
sec_per_frame = 1/frame_per_sec #32ms

#Extracting .wav, .txt file lists
data_dir = '/home/data/kyungsu/AMT'
sub_dir_list = ['AkPnBsdf', 'AkPnCGdD', 'AkPnStgb', 'ENSTDkAm', 'ENSTDkCl',
                'SptkBGAm', 'SptkBGCl', 'StbgTGd2']

wavfile_list = []
txtfile_list = []
for sub in sub_dir_list:
  txtfile_list = txtfile_list + \
                 sorted(glob.glob(os.path.join(data_dir,sub,'MUS','*.txt')))
  wavfile_list = wavfile_list + \
                 sorted(glob.glob(os.path.join(data_dir,sub,'MUS','*.wav')))


#Do some CQT
for i,(wavfile,txtfile) in enumerate(zip(wavfile_list,txtfile_list)):
    #Make CQT matrix
    wav = utils.load_wav(wavfile)
    cqt_wav = utils.cqt(wav)
    np.save(wavfile+".npy",cqt_wav)

    #Make labeled data
    y_data = np.zeros((88,cqt_wav.shape[1]))
    with open(txtfile) as f:
        lines=f.readlines()
    lines = lines[1:]
    lines = [line.strip().split('\t') for line in lines]
    for line in lines:
        start_frame = int(round(frame_per_sec*float(line[0])))
        end_frame = int(round(frame_per_sec*float(line[1])))
        pitch = int(line[2])-21
        for j in range(start_frame,end_frame):
            y_data[pitch,j]=1 asdf
    np.save(txtfile+".npy",y_data)
    print "%d / %d" % (i+1,len(txtfile_list))
