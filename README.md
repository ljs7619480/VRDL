# Kaggle_CS_T0828_HW1
Code for solution in Kaggle fine-grained image classification Challenge with modified Stanfor-Cars dataset.
https://www.kaggle.com/c/cs-t0828-2020-hw1/overview

## Hardware
The following specs were used to create the original solution.
- Ubuntu 18.04 LTS
- Intel(R) Core(TM) i5-9600K CPU @ 3.70GHz
- NVIDIA Corporation GP102 [GeForce GTX 1080 Ti] (rev a1) 

## Reproducing Submission
To reproduct my submission without retrainig, do the following steps:
1. [Installation](#installation)
2. [Dataset Preparation](#dataset-preparation)
3. [Download Pretrained models](#pretrained-models)
4. [Evaluate](#evaluate)
5. [Make Submission](#make-submission)
6. [Reference](#reference)

## Installation
All requirements should be detailed in requirements.txt.
```
# python version: Python 3.6.9
pip3 install -r requirements.txt
```

## Dataset Preparation
Please refer to the kaggle main page, download the dataset there, and rename the dataset folder as "PATH_TO_PROJECT/dataset".
After downloading images, running the following script to generate the label file required.
```
cd PATH_TO_PROJECT
python3 label_preprocess.py 
```

The data directory is now structured as:
```
dataset
  +- testing_data
  |  +- testing_data / testing_data
  +- training_data / training_data
  |  - labels_seri_map.csv
  |  - training_labels_seri.csv
  |  - training_labels.csv
```

## Training
In configs directory, you can find configurations I used train my final models. My final submission is ensemble of resnet34 x 5, inception-v3 and se-resnext50, but ensemble of inception-v3 and se-resnext50's performance is better.

### Train models
- Using the following script to get more information
```
$ python train.py --help
```

- Example
```
# training the restnet152 model with pretrained model
$ python3 train.py "dataset" --pretrained --arch retnet152 --batch-size 32
# training the same model starting from previous checkpoint(default path to model weight) with smaller learning rate
$ python3 train.py "dataset" --arch retnet152 --batch-size 32 --lr 0.0001 --resume "model_saved/checkpoint.pth"

```

### Pretrained models
TBA

## Evaluate
Add the --evalute flag will random pickup 3000 images from training_data with the following augmentation trick applied, and show the Top1, Top5 accuracy.
- RandomHorizontalFlip,
- RandomPerspective,
- Resize
- CenterCrop
- ColorJitter
- Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
```
python3 train.py "dataset" --arch retnet152 --batch-size 32 --resume "PATH_TO_MODEL_WEIGHT" --evaluate 
```

## Make Submission
Add the --test flat will generate the predict result in result.csv file.
```
python3 train.py "dataset" --arch retnet152 --batch-size 32 --resume "PATH_TO_MODEL_WEIGHT" --test
```

## reference
This project is modify from [pytorch
/
examples](
https://github.com/pytorch/examples/tree/42e5b996718797e45c46a25c55b031e6768f8440/imagenet)