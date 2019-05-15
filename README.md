# flower-classifier

A AI flower classifier

## Setup project
1. Run command __pip install -r requirements.txt__ to install the project dependencies.

## Training

* Basic usage:  __python train.py <YOUR_DATA_DIRECTORY>__

### Options:

* Set directory to save checkpoints: __python train.py <YOUR_DATA_DIRECTORY> --save_dir <YOUR_CHECKPOINT_DIRECTORY>__
* Choose architecture: __python train.py data_dir --arch "vgg"__
* Set hyperparameters: __python train.py  <YOUR_DATA_DIRECTORY> --learning_rate 0.01 --hidden_units 512 --epochs 20__
* Use GPU for training: __python train.py  <YOUR_DATA_DIRECTORY> --gpu__

## Predict

* Basic usage:  __python predict.py <YOUR_IMAGE_PATH> <YOUR_CHECKPOINT_PATH>__

### Options:

* Return top KK most likely classes: __python predict.py <YOUR_IMAGE_PATH> <YOUR_CHECKPOINT_PATH> --top_k 3__
* Use a mapping of categories to real names: __python predict.py  <YOUR_IMAGE_PATH> <YOUR_CHECKPOINT_PATH> --category_names cat_to_name.json__
* Use GPU for inference: __python predict.py  <YOUR_IMAGE_PATH> <YOUR_CHECKPOINT_PATH> --gpu__
