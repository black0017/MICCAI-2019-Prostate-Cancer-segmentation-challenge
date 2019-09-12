# MICCAI 2019 Prostate Cancer segmentation challenge
### Work in progress! Updates to come.
Attempt to process high resolution images in google collab.

This project is about Deep Learning in microscopy 2D high-resolution(5Kx5k pixels) image segmentation.
MICCAI 2019 Prostate Cancer segmentation challenge data were used.

Data can be downloaded from here: https://gleason2019.grand-challenge.org/

In order to reproduce the results of this challenge place the extracted data in a  google collab folder and use it as root path


## Usage
### 1. Open miccai.ipynb in Google Colab
1. Go to https://colab.research.google.com
2. **```File```** > **```Upload notebook...```** > **```GitHub```** > **```Paste this link:``` https://github.com/black0017/MICCAI-2019-Prostate-Cancer-segmentation-challenge/blob/master/MICCAI_2019.ipynb**
3. Ensure that **```Runtime```** > **```Change runtime type```** is ```Python 3``` with ```GPU```
### 2. Initial imports, install, initializations
Second step is to install all the required dependencies. Select the first and second code cells and push ```shift+enter```. You'll see running lines of executing code. Wait until it's done (1-2 minutes).
### 3. Helper functions

### 4. Read annotations and offline processing
Applies majority voting for the provided annotations to generate training labels
Executed only once due to poor time complexity (rougly 2-3 minutes to generate 1 image label)
