# 2021-22Term2 - AIST4010 Project
Author: Jason Chan Wing Kit

SID: 1155142198

### Folder Structure
- `Baseline`: Stores the model setup, training, and evaluation process of the **Spectrogram** learning approach
  - `densenet`: DenseNet121 baseline
  - `efficientnet`: EfficientNet-B2 baseline
  - `freqCNN`: LargeConv CNN in the paper, also includes some training results for the valence level predictions in code comments
  - `inception`: InceptionV3 baseline
  - `resnet`: ResNet34 baseline
  - `vgg`: VGG16 baseline
  - `results`: Contains the trained model for evaluation purposes, and the validation loss records during training

- `PANN`: Stores the model setup, training, eand evaluation process of the **Waveform** learning approach
  - `results`: Contains the trained model for evaluation purposes, and the validation loss records during training
  - `wavenet_valence.py`: For training models to predict valence level, includes some training results in code comments

- `data`, `helper`: Stores the functions used for dataset loading / preprocessing, and for training / evaluating

- `data_preparation`: Stores the IPYNB notebook used for dataset preparation
  - `data-fetch.ipynb`: Fetches the 30s preview clip of the song from Spotify API, requires OAUTH token
  - `gen-spectrogram.ipynb`: Converts the fetched data to waveforms and spectrogram images (takes long time)
  - `data-statistics.ipynb`: Plots the song genres and sentiment label distribution of the dataset
  - `plot-loss.ipynb`: Plots the validation loss curve during model training, requires the losses files
  - Others are for exploratory purposes during the model training

- `models`, `old_spec_approach`: Stores the src codes used in the earlier stage, but deprecated in the final report
  - `old_spec_approach\summary.txt`: Stores the results of some training results in the early stage

- `inference.ipynb`: For the inference of the trained models.
