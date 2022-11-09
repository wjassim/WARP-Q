

# Quality Prediction For Generative Neural Speech Codecs (WARP-Q)
This code is to run the WARP-Q speech quality metric.
https://github.com/WissamJassim/WARP-Q.git

WARP-Q is an objective, full-reference metric for perceived speech quality. It uses a subsequence dynamic time warping (SDTW) algorithm as a similarity between a reference (original) and a test (degraded) speech signal to produce a raw quality score. It is designed to predict quality scores for speech signals processed by low bit rate speech coders. 



- [News](#news)
- [Overview](#overview)
- [Using WARP-Q](#using-warp-q)
- [Score Mapping](#score-mapping)
- [Model Design](#model-design)
- [Citing](#citing)






## News
- November 2022: Adding a new API with two running modes via command line arguments and different pretrained mapping models - currently still under construction. Please bear with me. It will be done soon. 
- July 2022: A new manuscript entitled [“Speech quality assessmentwith WARP‐Q: From similarity to subsequence dynamic time warp cost”](https://ietresearch.onlinelibrary.wiley.com/doi/epdf/10.1049/sil2.12151) has been accepted for publication in the IET Signal Processing Journal. In this paper, we present the detailed design of WARP-Q with a comprehensive evaluation and analysis of the model components, design decisions, and salience of parameters to the model's performance. The paper also presents a comprehensive set of benchmarks with standard and new datasets and benchmarks against other standard and state-of-the-art full reference speech quality metrics. Furthermore, we compared WARP-Q results to the results from two state-of-the-art reference free speech quality measures. We also explored the possibility of mapping raw WAPR-Q scores onto target MOS using different machine learning (ML) models.

- Jan 2021: Publishing initial codes of WARP-Q.




## Overview

Speech coding has been shown to achieve good speech quality using either waveform matching or parametric reconstruction. For very low bit rate streams, recently developed generative speech models can reconstruct high quality wideband speech from the bit streams of standard parametric encoders at less than 3 kb/s. Generative codecs produce high quality speech based on synthesising speech from a DNN and the parametric input. 

The problem is that the existing objective speech quality models (e.g., ViSQOL, POLQA) cannot be used to accurately evaluate the quality of coded speech from generative models as they penalise based on signal differences not apparent in subjective listening test results. Motivated by this observation, we propose the WARP-Q metric, which is robust to low perceptual signal changes introduced by low bit rate neural vocoders. Figure 1 shows illustrates a block diagram of the proposed algorithm.    

<div align="center">

| <img src="Resources/WARP_Q_metric.png" width="700"> | 
|:--| 

Figure 1: High‐level block diagram of the WARP‐Q metric
</div>

The algorithm of WARP-Q metric consists of four processing stages:  

- Pre-processing: silent non-speech segments from reference and degraded signals are detected and removed using a voice activity detection (VAD) algorithm. 

- Feature extraction: <img src="https://render.githubusercontent.com/render/math?math=K"> cepstral coefficients of the reference and degraded signals are first generated. The obtained MFCCs representations are then normalised so that they have the same segmental statistics (zero mean and unit variance) using the cepstral mean and variance normalisation (CMVN). 

- Similarity comparison: WARP-Q uses the SDTW algorithm to estimate the similarity between the reference degraded signals in the MFCC domain. It first divides the normalised MFCCs of the degraded signal into a number, <img src="https://render.githubusercontent.com/render/math?math=L">, of patches. For each degraded patch <img src="https://render.githubusercontent.com/render/math?math=X">, the SDTW algorithm then computes the accumulated alignment cost between <img src="https://render.githubusercontent.com/render/math?math=X"> and the reference MFCC matrix <img src="https://render.githubusercontent.com/render/math?math=Y">. The computation of accumulated alignment cost is based on an accumulated alignment cost matrix <img src="https://render.githubusercontent.com/render/math?math=D_{(X,Y)}"> and its optimal path <img src="https://render.githubusercontent.com/render/math?math=P^\ast"> between <img src="https://render.githubusercontent.com/render/math?math=X"> and <img src="https://render.githubusercontent.com/render/math?math=Y">. Figure 2 shows an example of this stage. Further details are available in [[1]](https://ietresearch.onlinelibrary.wiley.com/doi/epdf/10.1049/sil2.12151) and [[2]](https://arxiv.org/pdf/2102.10449).   

<div align="center">

| ![subSeqDTW.png](Resources/subSeqDTW.png) | 
|:--| 

<div align="left">

Figure 2: SDTW-based accumulated cost and optimal path between two signals. (a) plots of a reference signal and its corresponding coded version from a WaveNet coder at 6 kb/s (obtained from the VAD stage), (b) normalised MFCC matrices of the two signals, (c) plots of SDTW-based accumulated alignment cost matrix <img src="https://render.githubusercontent.com/render/math?math=D_{(X,Y)}"> and its optimal path <img src="https://render.githubusercontent.com/render/math?math=P^\ast"> between the MFCC matrix <img src="https://render.githubusercontent.com/render/math?math=Y"> of the reference signal and a patch <img src="https://render.githubusercontent.com/render/math?math=X"> extracted from the MFCC matrix of the degraded signal. The optimal indices ( <img src="https://render.githubusercontent.com/render/math?math=a^{\ast}"> and  <img src="https://render.githubusercontent.com/render/math?math=b^{\ast}"> ) are also shown. <img src="https://render.githubusercontent.com/render/math?math=X"> corresponds to a short segment (2 s long) from the WaveNet signal (highlighted in green color). 
</div>

</div>


- Subsequence score aggregation: the final quality score is representd by a median value of all alignment costs. 


An evaluation using waveform matching, parametric and generative neural vocoder based codecs as well as channel and environmental noise shows that WARP-Q has better correlation and codec quality ranking for novel codecs compared to traditional metrics as well as the versatility of capturing other types of degradations, such as additive noise and transmission channel degradations. 

The results show that although WARP-Q is a simple model building on well established speech signal processing features and algorithms it solves the unmet need of a speech quality model that can be applied to generative neural codecs.


## Using WARP-Q

### Installation

To run the code, please implement the following steps:

1. Clone this repository:
    ```ruby
    git clone https://github.com/wjassim/WARP-Q.git
    ```
2. Create a new environment named warpq and then activate it: 
    ```ruby
    conda create --name warpq python=3.9
    conda activate warpq
    ```
3. Change working directory to the path of warpq repo and then install the dependencies:
    ```ruby
    cd path/of/clonned/warpq/repo
    pip install -r requirements.txt
    ```

### Prediction
There are two running modes (controlled by --mode argument) available to predict the quality of speech via command line arguments:

- predict_csv: predict quality scores of multi reference and degraded speech files listed in a csv file given by *--csv_input* argument. The results will be saved to a csv file given by *--csv_output* argument.
- predict_file: predict quality score between two speech files, reference file given by *--org* argument and its degraded given by *--deg* argument. The results will be printed to the console only.   

To predict the quality of all .wav files listed in a csv table, run WARP-Q with predict_csv mode:
 
```ruby
python warpq.py --mode predict_csv --csv_input /path/to/input/csv/file.csv --csv_output /path/to/results/file.csv --mapping_model /path/to/mapping/model/file.zip
```

Example: 
```ruby
python warpq.py --mode predict_csv --csv_input ./audio_samples.csv --csv_output ./results.csv --mapping_model ./models/RandomForest/Genspeech_TCDVoIP_PSup23.zip
```

To predict the quality of two .wav files, run WARP-Q with predict_file mode:
 
```ruby
python warpq.py --mode predict_file --org /path/to/original/speech/file.wav --deg /path/to/degraded/speech/file.wav --mapping_model /path/to/mapping/model/file.zip
```
Example: 
 
```ruby
python warpq.py --mode predict_file --org ./audio/p239_021.wav --deg ./audio/p239_021_evs.wav --mapping_model ./models/RandomForest/Genspeech_TCDVoIP_PSup23.zip
```
    
The provided code computes raw WARP-Q scores. It also maps them onto the standard MOS rating using a mapping model as explained in the following section. <!--As proposed in [2], there are different models available. More details about these models will be provided soon. -->

## Score Mapping
The original implementation of WARP-Q provides quality scores with negative correlations, i.e., lower rating means better quality, as this metric is designed based on  subsequence alignment costs of speech signals. To make WARP-Q scores compatible with that of other standard quality metrics, in [[1]](https://ietresearch.onlinelibrary.wiley.com/doi/epdf/10.1049/sil2.12151), we explored the possibility of mapping these scores onto standard MOS ratings (higher rating means better quality) using ML algorithms. Several models have been employed and evaluated.

In this repository, we provide mapping models based on two ML algorithms: 
- Deep neural networks with a sequential stack of three dense layers
- Random-forest-based regressor 

The idea here is to give users the opportunity to try different mapping models and select a model that is more suitable for their needs and data. To use any of these models, we need to provide its location to the code using *--mapping_model* argument. 


Each model was trained using different databases to test its performance under different signal distortions and background effects. Table 1 below provides details about each model with the employed databases.


<div align="center">
Table 1: file name and location of each mapping model with its corresponding database  

| Database | Purpose | File name of model 1| File name of model 2 |
| --------------- | --------------- | --------------- | ---------------|
| Genspeech | Assessing quality in low bit rate codecs | [./models/SequentialStack/<br />Genspeech.zip](https://github.com/wjassim/WARP-Q/blob/main/models/SequentialStack/Genspeech.zip)  | [./models/RandomForest/<br>Genspeech.zip](https://github.com/wjassim/WARP-Q/blob/main/models/RandomForest/Genspeech.zip)  |
| TCD‐VoIP | Assessing quality in VoIP applications | [./models/SequentialStack/<br>TCDVoIP.zip](https://github.com/wjassim/WARP-Q/blob/main/models/SequentialStack/TCDVoIP.zip) | [./models/RandomForest/<br>TCDVoIP.zip](https://github.com/wjassim/WARP-Q/blob/main/models/RandomForest/TCDVoIP.zip) |
| ITU‐T P. Supplement 23 (P.Sup23) | Interworking with wireless, transmission standards, and channel degradations|  [./models/SequentialStack/<br>PSup23.zip](https://github.com/wjassim/WARP-Q/blob/main/models/SequentialStack/PSup23.zip) |  [./models/RandomForest/<br>PSup23.zip](https://github.com/wjassim/WARP-Q/blob/main/models/RandomForest/PSup23.zip) |
| Genspeech + TCD-VoIP + P.Sup23 | Assessing overall quality | [./models/SequentialStack/<br>Genspeech_TCDVoIP_PSup23.zip](https://github.com/wjassim/WARP-Q/blob/main/models/SequentialStack/Genspeech_TCDVoIP_PSup23.zip) | [./models/RandomForest/<br>Genspeech_TCDVoIP_PSup23.zip](https://github.com/wjassim/WARP-Q/blob/main/models/RandomForest/Genspeech_TCDVoIP_PSup23.zip) |


</div>

As shown in the table, there are two zip files for each set of data. Each zip file contains a trained model with its data scaler. For example, the [./models/SequentialStack/Genspeech.zip](https://github.com/wjassim/WARP-Q/blob/main/models/SequentialStack/Genspeech.zip) file contains a sequential stack model that was trained using the Genspeech database to map its raw scores of WARP-Q onto Genspeech MOS ratings. The mentioned zip file also contains a data scaler corresponding to this data which can be used to scale unseen (test) data before applying the trained model for quality score prediction. 

To further expand the ability of this mapping technique to cover data with more scenarios of signal distortion and background effects, we provide models that were trained using a combination of the three databases, i.e., Genspeech + TCD-VoIP + P.Sup23. Links of these models are provided in the forth row of the table shown above.    

See our paper [[1]](https://ietresearch.onlinelibrary.wiley.com/doi/epdf/10.1049/sil2.12151) for more details about speech databases and mapping models used in this repository. 








## Model Design

Our previous work [[2]](https://arxiv.org/pdf/2102.10449) introduced the WARP-Q metric evaluating the performance for the chosen design parameters without evaluating or analysing their influence. In our new paper [[1]](https://ietresearch.onlinelibrary.wiley.com/doi/epdf/10.1049/sil2.12151), we establish the sensitivity and importance of model components and design choices to the overall metric performance. The purpose of the experiments presented in this Section was to establish a default set of parameters/settings for the proposed model. Furthermore, the experiments were conducted to find default WARP-Q settings that prioritise the Genspeech dataset but work as well as possible for other scenarios. 

The evaluated parameters are: 

  - Sampling frequency of input signals (8 kHz vs 16 kHz)
  - Spectral features (MFCC vs Melspectrogram)
  - Maximum frequency for spectral representation, (<img src="https://render.githubusercontent.com/render/math?math=f_{max}">=4, 5, 6, and 8 kHz)
  - Number of MFCC coefficients (<img src="https://render.githubusercontent.com/render/math?math=K">=12,13,16, and 24 )
  - Patch size for evaluation (0.2, 0.4, and 0.6 seconds)
  - Effect of VAD algorithm on quality scores
  - Aggregate function for temporal pooling of costs
  - Effect of DTW step size, <img src="https://render.githubusercontent.com/render/math?math=\Sigma">
 
Figure 3 compares the performance of the evaluated parameters. Please see section 4 of our new paper [[1]](https://ietresearch.onlinelibrary.wiley.com/doi/epdf/10.1049/sil2.12151) for more details about this figure, score distributions, and summary of best results.  
 
 <div align="center">
  
| <img src="Resources/Model_Design.jpg" width="750"> | 
|:--| 

<div align="left">
Figure 3: (a)-(g) Pearson and Spearman correlation coefficients for evaluated factors using the Genspeech, TCD-VoIP, P.Sup23 EXP1, and P.Sup23 EXP3 databases, (h) Pearson correlation coefficient as a function of Spearman correlation coefficient for different parameter values across datasets. Lower correlation coefficients indicate better results.    

</div >  
 
</div>

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
      pages={401-405},
      doi={10.1109/ICASSP39728.2021.9414901}
     }



