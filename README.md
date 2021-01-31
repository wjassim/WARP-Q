<p align="center">
    <img src="Resources/WARP_Q_metric.png" width="600">
    
    Blockgiagram of WAAR-Q metric
</p>

<p align="center">
    <img src="Resources/subSeqDTW.png" >
</p>

# WARP-Q Speech Quality Metric
This code is to run the WARP-Q speech quality metric.

https://github.com/WissamJassim/WARP-Q.git

WARP-Q (Quality Prediction For Generative Neural Speech Codecs) is an objective, full-reference metric for perceived speech quality. It uses a dynamic time warping (DTW) algorithm as a similarity between a reference (original) and a test (degraded) speech signal to produce a raw quality score.

# Requirements
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

# Run WARPQ_main_code.py

Input:

    - The main_test function calls a csv file that contains paths of audio files. 
    
    - The csv file cosists of four columns: 
    
        - Ref_Wave: reference speech
        
        - Test_Wave: test speech
        
        - MOS: subjective score (optinal, for plotting only)
        
        - Codec: type of speech codec for the test speech (optinal, for plotting only)
        
    
Output: 

    - Code will compute the WARP-Q quality scores between Ref_Wave and Test_Wave. It will then store the obrained results in a new column in the same csv file.  


# License



# Papers for citation

Design of the WARP-Q algorithm is described in detail in the following paper: 

W. A. Jassim, J. Skoglund, M. Chinen, and A. Hines, “WARP-Q: Quality prediction for generative neural speech codecs,” 2020, paper accepted for presenatation at the 2021 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP 2021). Date of acceptance: 30 Jan 2021
