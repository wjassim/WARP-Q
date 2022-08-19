

# Quality Prediction For Generative Neural Speech Codecs (WARP-Q)
This code is to run the WARP-Q speech quality metric.
https://github.com/WissamJassim/WARP-Q.git

WARP-Q is an objective, full-reference metric for perceived speech quality. It uses a subsequence dynamic time warping (SDTW) algorithm as a similarity between a reference (original) and a test (degraded) speech signal to produce a raw quality score. It is designed to predict quality scores for speech signals processed by low bit rate speech coders. 



- [News](#news)
- [General Description](#general-description)
- [Requirements](#requirements)
- [Demo running](#demo-running)
- [Citing](#citing)






## News
- July 2022: A new manuscript entitled [“Speech quality assessmentwith WARP‐Q: From similarity to subsequence dynamic time warp cost”](https://ietresearch.onlinelibrary.wiley.com/doi/epdf/10.1049/sil2.12151) has been accepted for publication in the IET Signal Processing Journal. In this paper, we present the detailed design of WARP-Q with a comprehensive evaluation and analysis of the model components, design decisions, and salience of parameters to the model's performance. The paper also presents a comprehensive set of benchmarks with standard and new datasets and benchmarks against other standard and state-of-the-art full reference speech quality metrics. Furthermore, we compared WARP-Q results to the results from two state-of-the-art reference free speech quality measures. We also explored the possibility of mapping raw WAPR-Q scores onto target MOS using different machine learning (ML) models.

- Jan 2021: Publishing initial codes of WARP-Q.




## General Description

Speech coding has been shown to achieve good speech quality using either waveform matching or parametric reconstruction. For very low bit rate streams, recently developed generative speech models can reconstruct high quality wideband speech from the bit streams of standard parametric encoders at less than 3 kb/s. Generative codecs produce high quality speech based on synthesising speech from a DNN and the parametric input. 

The problem is that the existing objective speech quality models (e.g., ViSQOL, POLQA) cannot be used to accurately evaluate the quality of coded speech from generative models as they penalise based on signal differences not apparent in subjective listening test results. Motivated by this observation, we propose the WARP-Q metric, which is robust to low perceptual signal changes introduced by low bit rate neural vocoders. Figure 1 shows illustrates a block diagram of the proposed algorithm.    

| <img src="Resources/WARP_Q_metric.png" width="700"> | 
|:--| 
| Figure 1: Blockgiagram of WARP-Q metric |

The algorithm of WARP-Q metric consists of four processing stages:  

- Pre-processing: silent non-speech segments from reference and degraded signals are detected and removed using a voice activity detection (VAD) algorithm. 

- Feature extraction: Mel frequency cepstral coefficients (MFCCs) representations of the reference and degraded signals are first generated. The obtained MFCCs representations are then normalised so that they have the same segmental statistics (zero mean and unit variance) using the cepstral mean and variance normalisation (CMVN). 

- Similarity comparison: WARP-Q uses the SDTW algorithm to estimate the similarity between the reference degraded signals in the MFCC domain. It first divides the normalised MFCCs of the degraded signal into a number, <img src="https://render.githubusercontent.com/render/math?math=L">, of patches. For each degraded patch <img src="https://render.githubusercontent.com/render/math?math=X">, the SDTW algorithm then computes the accumulated alignment cost between <img src="https://render.githubusercontent.com/render/math?math=X"> and the reference MFCC matrix <img src="https://render.githubusercontent.com/render/math?math=Y">. The computation of accumulated alignment cost is based on an accumulated alignment cost matrix <img src="https://render.githubusercontent.com/render/math?math=D_{(X,Y)}"> and its optimal path <img src="https://render.githubusercontent.com/render/math?math=P^\ast"> between <img src="https://render.githubusercontent.com/render/math?math=X"> and <img src="https://render.githubusercontent.com/render/math?math=Y">. Figure 2 shows an example of this stage. Further details are available in [1].   


| ![subSeqDTW.png](Resources/subSeqDTW.png) | 
|:--| 
| Figure 2: SDTW-based accumulated cost and optimal path between two signals. (a) plots of a reference signal and its corresponding coded version from a WaveNet coder at 6 kb/s (obtained from the VAD stage), (b) normalised MFCC matrices of the two signals, (c) plots of SDTW-based accumulated alignment cost matrix <img src="https://render.githubusercontent.com/render/math?math=D_{(X,Y)}"> and its optimal path <img src="https://render.githubusercontent.com/render/math?math=P^\ast"> between the MFCC matrix <img src="https://render.githubusercontent.com/render/math?math=Y"> of the reference signal and a patch <img src="https://render.githubusercontent.com/render/math?math=X"> extracted from the MFCC matrix of the degraded signal. The optimal indices ( <img src="https://render.githubusercontent.com/render/math?math=a^{\ast}"> and  <img src="https://render.githubusercontent.com/render/math?math=b^{\ast}"> ) are also shown. <img src="https://render.githubusercontent.com/render/math?math=X"> corresponds to a short segment (2 s long) from the WaveNet signal (highlighted in green color). |



- Subsequence score aggregation: the final quality score is representd by a median value of all alignment costs. 


An evaluation using waveform matching, parametric and generative neural vocoder based codecs as well as channel and environmental noise shows that WARP-Q has better correlation and codec quality ranking for novel codecs compared to traditional metrics as well as the versatility of capturing other types of degradations, such as additive noise and transmission channel degradations. 

The results show that although WARP-Q is a simple model building on well established speech signal processing features and algorithms it solves the unmet need of a speech quality model that can be applied to generative neural codecs.


## Requirements
Run using python 3.x and include these package dependencies in your virtual environment:

    - pandas 
    - librosa
    - numpy 
    - pyvad
    - skimage
    - speechpy
    - soundfile
    - scipy (optional)
    - seaborn (optional, for plotting only)
    - multiprocessing (optional, for parallel computing mode only)
    - joblib (optional, for parallel computing mode only)

## Demo running

- Run WARPQ_demo.py

Input:

    - The main_test function calls a csv file that contains paths of audio files. 
    
    - The csv file consists of four columns: 
    
        - Ref_Wave: reference speech
        - Test_Wave: test speech
        - MOS: subjective score (optinal, for plotting only)
        - Codec: type of speech codec (optinal, for plotting only)
            
Output: 

    - Code will compute the WARP-Q quality scores between Ref_Wave and Test_Wave. 
    - It will then store the obtained results in a new column in the same csv file.  


## Citing

Please cite our papers if you find this repository useful:

[[1] W. A. Jassim, J. Skoglund, M. Chinen, and A. Hines, “Speech quality assessmentwith WARP‐Q: From similarity to subsequence dynamic time warp cost,” IET Signal Processing, 1– 21 (2022)](https://ietresearch.onlinelibrary.wiley.com/doi/epdf/10.1049/sil2.12151)

    @article{Wissam_IET_Signal_Process2022,
      author = {Jassim, Wissam A. and Skoglund, Jan and Chinen, Michael and Hines, Andrew},
      title = {Speech quality assessment with WARP-Q: From similarity to subsequence dynamic time warp cost},
      journal = {IET Signal Processing},
      volume = {n/a},
      number = {n/a},
      pages = {},
      doi = {https://doi.org/10.1049/sil2.12151},
      url = {https://ietresearch.onlinelibrary.wiley.com/doi/abs/10.1049/sil2.12151},
      eprint = {https://ietresearch.onlinelibrary.wiley.com/doi/pdf/10.1049/sil2.12151},
     }


[[2] W. A. Jassim, J. Skoglund, M. Chinen, and A. Hines, “WARP-Q: Quality prediction for generative neural speech codecs,” ICASSP 2021 - 2021 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2021, pp. 401-405](https://arxiv.org/pdf/2102.10449)

    @INPROCEEDINGS{Wissam_ICASSP2021,
      author={Jassim, Wissam A. and Skoglund, Jan and Chinen, Michael and Hines, Andrew},
      booktitle={ICASSP 2021 - 2021 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
      title={Warp-Q: Quality Prediction for Generative Neural Speech Codecs}, 
      year={2021},
      volume={},
      number={},
      pages={401-405},
      doi={10.1109/ICASSP39728.2021.9414901}
     }



