# MICCAI 2019 Prostate Cancer segmentation challenge
### Work in progress! Updates to come.
Attempt to process high resolution images in google collab.

This project is about Deep Learning in microscopy 2D high-resolution(5Kx5k pixels) image segmentation.
MICCAI 2019 Prostate Cancer segmentation challenge data were used.

Data can be downloaded from here: https://gleason2019.grand-challenge.org/

In order to reproduce the results of this challenge place the extracted data in a  google collab folder and use it as root path.

Data loaders are availiable


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

### 5. Baseline experiment
The baseline approach:
1. Majority label Voting from different domain experts
2. Random shuffling 80% train 20% val split
3. 512x512x3 input patches as used in the original paper of 2d-Unet
4. Generate sample dataset rougly ~50 patches per input image
5. Train with Unet without data augmentation
6. Multi class dice loss functions will be used.

Hyperparameter tuning will *not* be applied is it is considered out of the scope of this assignment in this stage.

After the baseline expiriment further ideas/practices can be tested:

1. Split the dataset based on slice number and *not* randomly!
2. Apply Common data augmentation techniques
3. Examine input downsampling option
4. Use more recent model architectures and compare them to the baseline
5. Multiscale feature extraction would be a cool idea since image dimension is high

### 6. Current issues to encounter!!!
Even though is is shown that 25GB of RAM are available I could not store more than one inputs patche per image in memory.

I tried to save only the crop width and height but then then loader was really slow (4 sec to load,crop and preprocess the image  and 2,5 sec with 2 workers- which is still slow).

So I wrote the code to store only one patch per train image and changed the training patches every 50 subepochs.(similar to https://arxiv.org/abs/1804.02967)

The optimal solution as I see it now would be to store the preprocessed generated image patches in my drive and load at runtime!

There was also a problem with annotations that took me some time to figure out. The annotations were different from each expert. I used as a reference the Maps1 folder, because it had the same size(244 labels as the number of images). However, the annotated images were not excacly the same as the train images. Then I used as a reference the Maps5 folder and excluded the extra annotations.

## Support 
If you like this repo and find it useful, please consider (★) starring it, so that it can reach a broader audience.
