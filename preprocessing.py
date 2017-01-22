import librosa
import numpy as np
import scipy.io.wavfile
import os.path
import glob
import pdb
import matplotlib.pyplot as plt
#Parameters of CQT
target_sr=16000
hop_length=512
frame_per_sec = target_sr/float(hop_length) # 31.25
sec_per_frame = 1/frame_per_sec
n_bins = 264
bins_per_octave=36
real=False

#Extracting .wav, .txt file lists
data_dir = os.path.join(os.path.dirname(__file__),'data')
wavfile_list = sorted(glob.glob(os.path.join(data_dir,'*.wav')))
txtfile_list = sorted(glob.glob(os.path.join(data_dir,'*.txt')))
cqt_list = []
label_list = []

#Do some CQT
for i,(wavfile,txtfile) in enumerate(zip(wavfile_list,txtfile_list)):
    #Make CQT matrix
    original_sr,wav = scipy.io.wavfile.read(wavfile)
    wav = 0.5*(wav[:,0]+wav[:,1])
    wav_resample = librosa.core.resample(wav,original_sr,target_sr)
    cqt_wav=librosa.core.cqt(y=wav_resample,sr=target_sr,hop_length=hop_length,
                             fmin=librosa.core.note_to_hz('A0'),n_bins=n_bins,
                             bins_per_octave=bins_per_octave,real=real)
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
            y_data[pitch,j]=1
    np.save(txtfile+".npy",y_data)
    print "%d / %d" % (i+1,len(txtfile_list))

