# WARP-Q Speech Quality Metric
This code is to run the WARP-Q speech quality metric.

https://github.com/WissamJassim/WARP-Q.git

WARP-Q (Quality Prediction For Generative Neural Speech Codecs) is an objective, full-reference metric for perceived audio quality. It uses a dynamic time warping (DTW) measure of similarity between a reference (original) and a test (degraded) speech signal to produce a raw quality score.

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

# License

Use of this source code is governed by a Apache v2.0 license that can be found in the LICENSE file.

# Papers

Design of the WARP_Q algorithm is fully descibed in the follwing paper: 

W. A. Jassim, J. Skoglund, M. Chinen, and A. Hines, “WARP-Q: Quality prediction for generative neural speech codecs,” 2020, paper submitted to the 2021 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP).
