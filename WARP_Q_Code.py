
"""
WARP-Q: Quality Prediction For Generative Neural Speech Codecs

Run using python 3.x and include these package dependencies in your virtual environment:
    - pandas 
    - librosa
    - seaborn 
    - numpy 
    - scipy
    - pyvad
    - skimage
    - speechpy
    - soundfile 

@author: Dr Wissam Jassim
         University College Dublin
         wissam.a.jassim@gmail.com
         wissam.jassim@ucd.ie
         November 28, 2020 

"""


# Load libraries
import pandas as pd
import librosa, librosa.core, librosa.display
import seaborn as sns
import numpy as np
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from pyvad import vad, trim, split
from skimage.util.shape import view_as_windows
import speechpy
import soundfile as sf


################################ WARP-Q #######################################
def compute_WAPRQ(ref_path,test_path,sr=16000,n_mfcc=13,fmax=5000,patch_size=0.4,
                  sigma=np.array([[1,0],[0,3],[1,3]])):
# def compute_WAPRQ(ref_path,test_path,sr=16000,n_mfcc=13,fmax=5000,patch_size=0.4,
#                   sigma=np.array([[1, 1], [3, 2], [1, 3]])):   
    # Inputs:
    # refPath: path of reference speech
    # disPath: path pf degraded speech
    # sr: sampling frequency, Hz
    # n_mfcc: number of MFCCs
    # fmax: cutoff frequency
    # patch_size: size of each patch in s
    # sigma: step size conditon for DTW

    # Output:
    # WARP-Q quality score between refPath and disPath


    ####################### Load speech files #################################
    # Load Ref Speech
    if ref_path[-4:] == '.wav':
        speech_Ref, sr_Ref = librosa.load(ref_path,sr=sr) 
    else:
        if ref_path[-4:] == '.SRC': #For ITUT database if applicable
            speech_Ref, sr_Ref  = sf.read(ref_path, format='RAW', channels=1, samplerate=16000,
                           subtype='PCM_16', endian='LITTLE')
            if sr_Ref != sr:
                speech_Ref = librosa.resample(speech_Ref, sr_Ref, sr)
                sr_Ref = sr
        
    # Load Coded Speech
    if test_path[-4:] == '.wav':
        speech_Coded, sr_Coded = librosa.load(test_path,sr=sr)
    else: 
        if test_path[-4:] == '.OUT': #For ITUT database if applicable
            speech_Coded, sr_Coded  = sf.read(test_path, format='RAW', channels=1, samplerate=16000,
                           subtype='PCM_16', endian='LITTLE')
            if sr_Coded != sr:
                speech_Coded = librosa.resample(speech_Coded, sr_Coded, sr)
                sr_Coded = sr
    
    if sr_Ref != sr_Coded:
        raise ValueError("Reference and degraded signals should have same sampling rate!")
    
    # Make sure amplitudes are in the range of [-1, 1] otherwise clipping to -1 to 1 
    # after resampling (if applicable). We experienced this issue for TCD-VOIP database only
    speech_Ref[speech_Ref>1]=1.0
    speech_Ref[speech_Ref<-1]=-1.0
    
    speech_Coded[speech_Coded>1]=1.0
    speech_Coded[speech_Coded<-1]=-1.0
    
    ###########################################################################
   
    win_length = int(0.032*sr) #32 ms frame
    hop_length = int(0.004*sr) #4 ms overlap
    #hop_length = int(0.016*sr)
    
    n_fft = 2*win_length
    lifter = 3 
    
    # DTW Parameters
    Metric = 'euclidean'
        
    # VAD Parameters
    hop_size_vad = 30
    sr_vad = sr
    aggresive = 0
    
    # VAD for Ref speech
    vact1 = vad(speech_Ref, sr, fs_vad = sr_vad, hop_length = hop_size_vad, vad_mode=aggresive)
    speech_Ref_vad = speech_Ref[vact1==1]
    
    # VAD for Coded speech
    vact2 = vad(speech_Coded, sr, fs_vad = sr_vad, hop_length = hop_size_vad, vad_mode=aggresive)
    speech_Coded_vad = speech_Coded[vact2==1]
   
    # Compute MFCC features for the two signals
    
    # mfcc_Ref = librosa.feature.mfcc(speech_Ref_vad,sr=sr,n_mfcc=n_mfcc,fmax=fmax,
    #                                 n_fft=n_fft,win_length=win_length,hop_length=hop_length,lifter=lifter)
    # mfcc_Coded = librosa.feature.mfcc(speech_Coded_vad,sr=sr,n_mfcc=n_mfcc,fmax=fmax,
    #                                 n_fft=n_fft,win_length=win_length,hop_length=hop_length,lifter=lifter)
    
    mfcc_Ref = librosa.feature.melspectrogram(speech_Ref_vad,sr=sr,n_mels=n_mfcc,fmax=fmax,
                                    n_fft=n_fft,win_length=win_length,hop_length=hop_length)
    mfcc_Coded = librosa.feature.melspectrogram(speech_Coded_vad,sr=sr,n_mels=n_mfcc,fmax=fmax,
                                    n_fft=n_fft,win_length=win_length,hop_length=hop_length)
    
    # Feature Normalisation using CMVNW method 
    mfcc_Ref = speechpy.processing.cmvnw(mfcc_Ref.T,win_size=201,variance_normalization=True).T
    mfcc_Coded = speechpy.processing.cmvnw(mfcc_Coded.T,win_size=201,variance_normalization=True).T
    
    # Divid MFCC features of Coded speech into patches
    cols = int(patch_size/(hop_length/sr))
    window_shape = (np.size(mfcc_Ref,0), cols)
    step  = int(cols/2)
    
    mfcc_Coded_patch = view_as_windows(mfcc_Coded, window_shape, step)

    Acc =[]
    band_rad = 0.25  
    weights_mul=np.array([1, 1, 1])
     
    # Compute alignment cose between each patch and Ref MFCC
    for i in range(mfcc_Coded_patch.shape[1]):    
        
        patch = mfcc_Coded_patch[0][i]
        
        D, P = librosa.sequence.dtw(X=patch, Y=mfcc_Ref, metric=Metric, 
                                    step_sizes_sigma=sigma, weights_mul=weights_mul, 
                                    band_rad=band_rad, subseq=True, backtrack=True)
  
        P_librosa = P[::-1, :]
        b_ast = P_librosa[-1, 1]
        
        Acc.append(D[-1, b_ast] / D.shape[0])
        
    # Final score
    return np.median(Acc)



###############################################################################
############### #  Main Demo Test Function ####################################
###############################################################################

def main_test():

    # Load path of speech files stored in a csv file 
    # The csv files cosists of four columns: Ref_Wave	Test_Wave 	MOS 	Codec   
    All_Data = pd.read_csv('134369173_Wissam.csv',index_col=None)
    
    WARP_Q = [] # List to add WARP-Q scores
    # Run WARP-Q for each row
    for index, row in All_Data.iterrows():
        score = compute_WAPRQ(ref_path=row['Ref_Wave'],test_path=row['Test_Wave'])
        WARP_Q.append(score)
    
        print(row['Test_Wave'])
    
    
    # Add computed score to the same csv file
    All_Data['WARP-Q'] = WARP_Q
    
    
    # Plot WARP-Q scores per condition as a function of codec type
    dataX = All_Data.groupby('Codec').agg({'WARP-Q': 'mean','MOS':'mean'})
    dataX = dataX.reset_index()
    
    # Compute Pearson correlation coefficient
    pearson_coef, p_value = pearsonr(dataX['WARP-Q'], dataX.MOS)
    Spearmanr_coef, pval_spearman = spearmanr(dataX['WARP-Q'], dataX.MOS)
    
    sns.lmplot(x="MOS", y='WARP-Q', fit_reg=False, hue='Codec',
               data=dataX).fig.suptitle('Per-condition: Pearsonr= '+
                                        str(round(pearson_coef,2))+', Spearman='+str(round(Spearmanr_coef,2)))
    
                        
    pearson_coef, p_value = pearsonr(All_Data['WARP-Q'], All_Data.MOS)
    Spearmanr_coef, pval_spearman = spearmanr(All_Data['WARP-Q'], All_Data.MOS)                                  
    #plt.figure()           
    sns.relplot(x="MOS", y="WARP-Q", hue="Codec",
                palette="muted",data=All_Data).fig.suptitle('Per-sample: Pearsonr= '+
                                        str(round(pearson_coef,2))+', Spearman='+str(round(Spearmanr_coef,2)))
    

if __name__ == '__main__':
    main_test()
