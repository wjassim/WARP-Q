import librosa, librosa.core, librosa.display
import pandas as pd
import numpy as np
from pyvad import vad #, trim, split
from skimage.util.shape import view_as_windows
import speechpy
import soundfile as sf
import os
import zipfile
import tempfile
import pickle
import keras

class warpqMetric(object):
    '''
    warpqModel: Main class of WARP-Q metric to estimate quality scores. 
    It contains the required subfunctions and evaluation process.                                                
    '''  

    def __init__(self, args):
        ''' 
        Initialize the object’s state
        Inputs:
        1) The self
        2) args: 
            - mode: predict_csv or predict_file
            - csv_input: input csv file name for predict_csv mode
            - csv_ourput: output csv file name for predict_csv mode
            - org: original speech file for predict_file mode
            - deg: degraded speech file for predict_file mode
            - sr: sampling frequency, Hz
            - n_mfcc: number of MFCCs
            - fmax: cutoff frequency
            - patch_size: size of each patch in seconds
            - sigma: step size conditon for DTW 
            - apply_vad: condition for using vad algorithm  
            - mapping_model: file name of pretrained model for score mapping
        '''
        
        self.args = args
        
        if not self.args['sr'] in (8000, 16000):
            print("Sampling rate of audio files should be either 8 kHz or 16 kHz.")
            exit()
            
        # MFCC and DTW parameters
        self.win_length = int(0.032*self.args['sr']) #32 ms frame
        self.hop_length = int(0.004*self.args['sr']) #4 ms overlap
        #self.hop_length = int(0.016*self.sr)
        self.dtw_metric = 'euclidean'
        self.n_fft = 2*self.win_length
        self.lifter = 3 
        
        # VAD Parameters
        self.hop_size_vad = 30
        self.sr_vad = self.args['sr']
        self.aggresive = 0
        
        # some prints
        print('\n+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
        print('    WARP-Q: Quality Prediction For Generative Neural Speech Codecs       ')
        print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
        print('\nRun WARP-Q metric with:')
        print('- sr = ' + str(self.args['sr']) + ' Hz')
        print('- ' + str(self.args['mode']) + ' mode')
    
        # Load mapping model
        print('- mapping raw quality scores onto MOS using a ML model')
        self.load_model(args['mapping_model'])
        
           
    def load_model(self, zip_path):
        ''' 
        Load trained mapping model with its data scaler 
        '''
        print('\nLoading mapping model and data scaler from ' + zip_path + ':')
        
        # Create a temp dir
        with tempfile.TemporaryDirectory() as tmpdirname:
            
            # Extract the zip file
            with zipfile.ZipFile(zip_path) as thezip:
                thezip.extractall(tmpdirname)
                #print('The zip file of the model has been extracted in a temp directory')
            
            # Read model
            isH5Exist = os.path.exists(os.path.join(tmpdirname, 'mapping_model.h5'))
            
            if isH5Exist:
                try: 
                    self.mapping_model =  keras.models.load_model(os.path.join(tmpdirname, 'mapping_model.h5'))
                except Exception as e:
                    print(e)
                    exit()
                self.mapping_model_type = 'dnn_sequential'
                print('Model is based on deep neural networks with a sequential stack regressor from Keras. It is trained using ' + os.path.basename(zip_path)[0:-4] + ' database \n')
                self.mapping_model.summary()
                print('\n')
            
            else:
                try: 
                    self.mapping_model =  pickle.load(open(os.path.join(tmpdirname, 'mapping_model.pkl'), 'rb'))
                except Exception as e:
                    print(e)
                    exit()
                self.mapping_model_type = 'random_forest'
                print("Model is based on a random forest regressor from Sklearn. It is trained using " + os.path.basename(zip_path)[0:-4] + ' database \n')
                #print('\n')
            
            try: # Read data scaler
                self.data_scaler = pickle.load(open(os.path.join(tmpdirname, 'data_scaler.pkl'), 'rb'))
            except Exception as e:
                print(e)
                exit()
            
            
    def load_audio(self, ref_path, test_path):
        ''' Load speech files '''
        
        # 1) Load reference audio
        if ref_path[-4:] == '.wav':
            speech_Ref, sr_Ref = librosa.load(ref_path, sr=self.args['sr']) 
        else:
            if ref_path[-4:] == '.SRC': #For ITUT database if applicable
                speech_Ref, sr_Ref  = sf.read(ref_path, format='RAW', channels=1, samplerate=16000,
                               subtype='PCM_16', endian='LITTLE')
                if sr_Ref != self.args['sr']:
                    speech_Ref = librosa.resample(speech_Ref, sr_Ref, self.args['sr'])
            
        # 2) Load coded audio
        if test_path[-4:] == '.wav':
            speech_Coded, sr_Coded = librosa.load(test_path,sr=self.args['sr'])
        else: 
            if test_path[-4:] == '.OUT': #For ITUT database if applicable
                speech_Coded, sr_Coded  = sf.read(test_path, format='RAW', channels=1, samplerate=16000,
                               subtype='PCM_16', endian='LITTLE')
                if sr_Coded != self.args['sr']:
                    speech_Coded = librosa.resample(speech_Coded, sr_Coded, self.args['sr'])
        
               
        # Make sure amplitudes are in the range of [-1, 1] otherwise clipping to -1 to 1 
        # after resampling (if applicable). We experienced this issue for the TCD-VOIP database only
        speech_Ref[speech_Ref>1]=1.0
        speech_Ref[speech_Ref<-1]=-1.0
        
        speech_Coded[speech_Coded>1]=1.0
        speech_Coded[speech_Coded<-1]=-1.0
        
        return speech_Ref, speech_Coded


    def compute_alignment_cost(self, patch, mfcc_Ref):
        ''' 
        Compute the alignment cost between two spectral representations using 
        Subsequence DTW. For more details, please see Subsection 3.3 of our paper 
        "W. A. Jassim, J. Skoglund, M. Chinen, and A. Hines, “Speech quality assessment
        with WARP‐Q: From similarity to subsequence dynamic time warp cost,
        IET Signal Processing, 1– 21 (2022)". Available on: 
        https://ietresearch.onlinelibrary.wiley.com/doi/pdfdirect/10.1049/sil2.12151 
        '''
        
        D, P = librosa.sequence.dtw(X = patch, 
                                    Y = mfcc_Ref, 
                                    metric = self.dtw_metric, 
                                    step_sizes_sigma = self.args['sigma'], 
                                    weights_mul = np.array([1, 1, 1]), 
                                    band_rad = 0.25, 
                                    subseq = True, 
                                    backtrack = True)        
        P_librosa = P[::-1, :]
        b_ast = P_librosa[-1, 1]
        
        return D[-1, b_ast] / D.shape[0]
        
    
    def get_feaures_for_score_mapping(self, Acc):
        ''' 
        Extract features from alignment costs vector to map raw WARP-Q scores 
        onto MOS. For more details, please see Section 8 of our paper 
        "W. A. Jassim, J. Skoglund, M. Chinen, and A. Hines, “Speech quality assessment
        with WARP‐Q: From similarity to subsequence dynamic time warp cost,
        IET Signal Processing, 1– 21 (2022)". Available on: 
        https://ietresearch.onlinelibrary.wiley.com/doi/pdfdirect/10.1049/sil2.12151 
        '''
        
        Acc_df = pd.DataFrame({'Acc':Acc})
        
        Acc_fea = dict()
        Acc_fea['warpq_count']    = Acc_df['Acc'].count()
        Acc_fea['warpq_mean']     = Acc_df['Acc'].mean()
        Acc_fea['warpq_median']   = Acc_df['Acc'].median()
        Acc_fea['warpq_var']      = Acc_df['Acc'].var()
        Acc_fea['warpq_std']      = Acc_df['Acc'].std()
        Acc_fea['warpq_min']      = Acc_df['Acc'].min()
        Acc_fea['warpq_max']      = Acc_df['Acc'].max()
        
        quantile = Acc_df.quantile([.25,.5,.75])
        
        Acc_fea['warpq_25%']      = quantile.iloc[0,0]
        Acc_fea['warpq_50%']      = quantile.iloc[1,0]
        Acc_fea['warpq_75%']      = quantile.iloc[2,0]
        Acc_fea['warpq_skewness'] = Acc_df['Acc'].skew()
        Acc_fea['warpq_kurtosis'] = Acc_df['Acc'].kurtosis()
        
        return pd.DataFrame.from_dict([Acc_fea])
    
    
    
    def evaluate(self, ref_path, test_path):
        ''' 
        Compute WARP-Q score between two input speech signals
        Inputs:
        1) The self
        2) refPath: path of reference speech
        3) disPath: path pf degraded speech 

        Output:
        WARP-Q quality score between refPath and disPath 
        '''

        # Load speech files
        speech_Ref, speech_Coded = self.load_audio(ref_path, test_path)

        if self.args['apply_vad']:
            # VAD for Ref speech
            vact1 = vad(speech_Ref, 
                        self.args['sr'], 
                        fs_vad = self.sr_vad, 
                        hop_length = self.hop_size_vad, 
                        vad_mode = self.aggresive)
            
            speech_Ref = speech_Ref[vact1==1]
            
            # VAD for Coded speech
            vact2 = vad(speech_Coded, 
                        self.args['sr'], 
                        fs_vad = self.sr_vad, 
                        hop_length = self.hop_size_vad, 
                        vad_mode = self.aggresive)
            
            speech_Coded = speech_Coded[vact2==1]
       
        
        # Compute MFCC features for the two signals
        mfcc_Ref = librosa.feature.mfcc(y = speech_Ref,
                                        sr = self.args['sr'],
                                        n_mfcc = self.args['n_mfcc'],
                                        fmax = self.args['fmax'],
                                        n_fft = self.n_fft,
                                        win_length = self.win_length,
                                        hop_length = self.hop_length,
                                        lifter = self.lifter)
        
        mfcc_Coded = librosa.feature.mfcc(y = speech_Coded,
                                          sr = self.args['sr'],
                                          n_mfcc = self.args['n_mfcc'],
                                          fmax = self.args['fmax'],
                                          n_fft = self.n_fft,
                                          win_length = self.win_length,
                                          hop_length = self.hop_length,
                                          lifter = self.lifter)
        
        # Feature Normalisation using CMVNW method 
        mfcc_Ref = speechpy.processing.cmvnw(mfcc_Ref.T,
                                             win_size = 201,
                                             variance_normalization = True).T
        
        mfcc_Coded = speechpy.processing.cmvnw(mfcc_Coded.T,
                                               win_size = 201,
                                               variance_normalization = True).T
        
        # Divid MFCC features of Coded speech into patches
        cols = int(self.args['patch_size']/(self.hop_length/self.args['sr']))
        window_shape = (np.size(mfcc_Ref,0), cols)
        step  = int(cols/2)
        
        mfcc_Coded_patch = view_as_windows(mfcc_Coded, window_shape, step)

        Acc =[]
        #band_rad = 0.25  
        #weights_mul = np.array([1, 1, 1])
         
        # Compute alignment cost between each patch and Ref MFCCs        
        for i in range(mfcc_Coded_patch.shape[1]):    
            
            patch = mfcc_Coded_patch[0][i]
            score = self.compute_alignment_cost(patch, mfcc_Ref)
            Acc.append(score)  
        
        # Raw quality score
        rawScore = round(np.median(Acc), 3) #Eq. 3 in [1] 
        
        # Map raw score onto MOS
        fea = self.get_feaures_for_score_mapping(Acc)
        fea_scaled = self.data_scaler.transform(fea)
        
        if self.mapping_model_type == 'dnn_sequential':
            prediction  = self.mapping_model.predict(fea_scaled, verbose=0)
        else:
            prediction  = self.mapping_model.predict(fea_scaled)
            
        # remove unstable values
        mappedScore = round(prediction.item(), 3)
        if mappedScore > 5:
            mappedScore = 5
        if mappedScore < 1:
            mappedScore = 1
        
        # Return scores
        return rawScore, mappedScore
 
