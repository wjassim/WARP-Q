"""

WARP-Q: Quality Prediction For Generative Neural Speech Codecs

This code is to run the WARP-Q speech quality metric described in our papers:
    
[1] W. A. Jassim, J. Skoglund, M. Chinen, and A. Hines, “Speech quality assessment
with WARP‐Q: From similarity to subsequence dynamic time warp cost,” 
IET Signal Processing, 1– 21 (2022)

[2] W. A. Jassim, J. Skoglund, M. Chinen, and A. Hines, “WARP-Q: Quality prediction 
for generative neural speech codecs,” ICASSP 2021 - 2021 IEEE International 
Conference on Acoustics, Speech and Signal Processing (ICASSP), 2021, pp. 401-405
    

Warning: While this code has been tested and commented giving invalid input 
files may cause unexpected results and will not be caught by robust exception
handling or validation checking. It will just fail or give you the wrong answer.


Dr Wissam Jassim
wissam.a.jassim@gmail.com
November 7, 2022

"""

# Load libraries
from WARPQ.WARPQmetric import warpqMetric
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
import argparse
import os
from tqdm import tqdm


'''
###############################################################################
######################  Main Test Function ####################################
###############################################################################
'''

parser = argparse.ArgumentParser()

parser.add_argument('--mode', type=str, help='Either predict_file or predict_csv')
parser.add_argument('--csv_input', type=str, #default='./audio_paths.csv', 
                    help='''Path of a csv file which contains 
                    paths and info of the input audio files. The csv file consists of four columns: 
                        - Ref_Wave: path of reference (original) audio file 
                        - Test_Wave: path of test (degraded) audio file 
                        - MOS: subjective rating score (optional, for plotting only)
                        - Codec: type of speech codec, condition, or noise type (optional, for plotting only) 
                    See ./audio_samples.csv as an example of this file.''')  
parser.add_argument('--org', type=str, help='Path of the original (reference) speech file') 
parser.add_argument('--deg', type=str, help='Path of the degraded (processed) speech file')   
parser.add_argument('--sr', type=int, default=16000, help='Sampling frequency of speech signals in Hz. Only two sr values are currently supported: 16000 and 8000 Hz')                    
parser.add_argument('--n_mfcc', type=int, default=13, help='Number of MFCCs')                    
parser.add_argument('--fmax', type=int, default=5000, help='Cutoff frequency for MFCC in Hz')                    
parser.add_argument('--patch_size', type=float, default=0.4, help='Size of MFCC patch in seconds')
parser.add_argument('--sigma', type=list, default=[[1,0],[0,3],[1,3]], help='Step size conditon for DTW') 
parser.add_argument('--apply_vad', type=bool, default=True, help='Condition for using vad algorithm') 
parser.add_argument('--mapping_model', type=str, required=True, help='File name of pretrained model to map raw WARP-Q scores onto MOS')
parser.add_argument('--csv_output', type=str, help='Path and name of a csv file to save WARP-Q results')
parser.add_argument('--getPlots', type=bool, default=True, help='To plot the predicted scores vs MOS. If True, MOS and Codec type should be provided in the input csv file')                 

args = parser.parse_args()
args = vars(args)

if args['mode'] == 'predict_csv':
    if args['csv_input'] is None:
        raise ValueError('--csv_file argument with input csv file name is required')
    if args['csv_output'] is None:
        raise ValueError('--csv_output argument with output csv file name is required')

elif args['mode'] == 'predict_file':
    if args['org'] is None:
        raise ValueError('--org argument with path to input original speech file is required')
    if args['deg'] is None:
        raise ValueError('--deg argument with path to input degraded speech file is required') 
        
else:
    raise NotImplementedError('--mode argument given is not available')



def main(args):
    
    # Object of WARP-Q class
    warpq = warpqMetric(args)
    warpq_rawScore = [] # List to add WARP-Q scores
    warpq_mappedScore = []
    
    if args['mode'] == 'predict_csv':
    
        # Load path of speech files stored in a csv file 
        # The csv file consists of data with four columns: Ref_Wave, Test_Wave, MOS, and Codec  
        df = pd.read_csv(args['csv_input'], index_col=None)
        
        # Iterative process
        for index, row in tqdm(df.iterrows(), total = df.shape[0], desc="Compute quality sores..."):
            
            rawScore, mappedScore = warpq.evaluate(ref_path = row['Ref_Wave'], test_path = row['Test_Wave'])
            
            warpq_rawScore.append(rawScore)
            warpq_mappedScore.append(mappedScore)
        
        # Add computed score to the same csv file
        df['Raw WARP-Q Score'] = warpq_rawScore
        df['Mapped WARP-Q Score'] = warpq_mappedScore
        
        # Save the results
        if not os.path.exists(os.path.dirname(args['csv_output'])):
            os.makedirs(os.path.dirname(args['csv_output']))
        
        df.to_csv(args['csv_output'], index = None)
                                              
        if args['getPlots']:
            
            # Compute per-sample Pearsonr and Spearmanr correlation coefficients for raw scores
            pearson_coef, p_pearson = pearsonr(df['Raw WARP-Q Score'], df['MOS'])
            Spearmanr_coef, p_spearman = spearmanr(df['Raw WARP-Q Score'], df['MOS'])                                  
             
            sns.relplot(x="MOS", y="Raw WARP-Q Score", hue="Codec", palette="muted",
                        data=df).fig.suptitle('Correlations: Pearsonr= '+ str(round(pearson_coef,2)) +
                            ', Spearman='+str(round(Spearmanr_coef,2)))
            
            # Compute per-sample Pearsonr and Spearmanr correlation coefficients for mapped scores
            pearson_coef, p_value = pearsonr(df['Mapped WARP-Q Score'], df['MOS'])
            Spearmanr_coef, p_spearman = spearmanr(df['Mapped WARP-Q Score'], df['MOS'])                                  
            
            sns.relplot(x="MOS", y="Mapped WARP-Q Score", hue="Codec", palette="muted",
                        data=df).fig.suptitle('Correlations: Pearsonr= '+ str(round(pearson_coef,2)) +
                            ', Spearman='+str(round(Spearmanr_coef,2)))
        
        print('\nResults are saved in ' + args['csv_output']) 
        
    else: #predict_file mode
    
        print("Compute quality sores...")
        warpq_rawScore, warpq_mappedScore = warpq.evaluate(args['org'], args['deg'])
        
        print('\nRaw WARP-Q score (lower rating means better quality): ' + str(warpq_rawScore)) 
        print('Mapped WARP-Q score (higher rating means better quality): ' + str(warpq_mappedScore))  
        
    print('Done!')
    
    
if __name__ == '__main__':
    
    main(args)
    
    