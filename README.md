# MICCAI 2019 Prostate Cancer segmentation challenge
This repo segments high-resolution images.

Google colab notebook is also [available](https://colab.research.google.com/drive/1biPE5drh_3TPMraykW2taJ7v7Gx3FuBx?usp=sharing).

This project is about Deep Learning in microscopy images.

The images are 2D but really high resolution ~5Kx5k pixels.

There are multiple annotators that manually segment the images.
Each annotator segments different number of images.

We need to map the annotators to the image and perform the so-called majority voting to generate the labels.

MICCAI 2019 Prostate Cancer segmentation [challenge data](https://gleason2019.grand-challenge.org/) were used.
The data are relativly easy to download, but you need to make an account.

### Installation step

Clone the project, create a new virtual environment, and run:
```
pip install -r requirements.txt
```

### Data
Download the data and place them in the project folder.
I named the folder "MICCAI_2019_pathology_challenge"


#### Read annotations and offline processing
Applies majority voting for the provided annotations to generate training labels
Executed only once due to poor time complexity: 
roughly <1 minute to generate 1 image label

Check the path name in the generate_labels.py script and run it:

```
python generate_labels.py
```
It takes ~ 2 hours, so take a break and enjoy your coffe ;) !

#### 5. Baseline experiment 

After checking the paths, run:
```
python train.py
```

The baseline approach:
1. Majority label Voting from different domain experts
2. Random shuffling 80% train 20% val split
3. 512x512x3 input patches
4. Unet architecture
4. Generate 30 samples per train image and 10 per val img
5. Train with Unet without data augmentation
6. Multi class dice loss functions will be used.


After the baseline experiment further ideas/practices can be tested:

1. Split the dataset based on slice number and ID and not randomly!
2. Apply common data augmentation techniques
3. Examine input down-sampling option
4. Use more recent model architectures and compare them to the baseline



## Medical Zoo Pytorch
For medical imaging projects, visit  [Medical Zoo pytorch](https://github.com/black0017/MedicalZooPytorch "MedZoo") project. 



## Support 
If you like this repo and find it useful, please consider (★) starring it, so that it can reach a broader audience.
