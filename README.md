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

Workflow:
1. DataCleaning/Processing
    1. DataSet Splitting -> 90 : 10
    1. For Training Set: 90%
        1. Possibility of removing Watermarks?
        1. Padding To Square(resizing),
        1. DownSizing images (to reduce memory requirements)
        1. Normalizing Channels (for faster Learning)
    1. For Test Set: 10%
        1. No Augmentation, only resizing(Padding to Square) to fit the model inputs. use it raw to test for accuracy and generalization
1. Model Building/Training
    1. Importing Model from torchvision
    1. Freezing of all layers except the classification layer.
    1. Train Model and Assess performance
        1. Epoch, LR, Loss Fn, Additional Metrics


        1. IF BAD: Modify/Reselect Model
        1. IF GOOD: Progressive Unfreezing of Params for deeper learning of features. 

