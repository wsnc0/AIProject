# AIProject

Project Description

CNN Classifier for dataset: https://www.kaggle.com/datasets/ismailpromus/skin-diseases-image-dataset/data

\Proj Folder
| - \data
|   | - Class 0
|   | - Class 1
|   | - Class 2
|   | - Class 3
|   | - Class 4
|   | - Class 5
|   | - Class 6
|   | - Class 7
|   | - Class 8
|   | - Class 9
| - dataProcessing.py
| - README.MD
| - test1.ipynb

Task: Supervised Classification task with a CNN, using Transfer Learning on a EfficientNet Model

Objectives:
1. DataCleaning/Processing
    1. DataSet Splitting -> 90 : 10
    1. For Training Set: 90%
        1. Possibility of removing Watermarks?
        1. Padding To Square,
        1. DownSizing images (to reduce memory requirements)
        1. Normalizing Channels (for faster Learning)
    1. For Test Set: 10%
        1. No Augmentation or anything, use it raw to test for accuracy and generalization
1. Model Building/Setup
    1. Importing Model from torchvision
    1. Freezing of all layers except the classification layer.
    1. 
1. Model Training
    1. Number of Epoch, LR, BatchSize,  
    1. Progressive Unfreezing of Params for deeper learning of features. 
1. Model Assessment
    1. Additional Metrics:
