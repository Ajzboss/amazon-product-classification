# Investigating Deep Learning Models for Amazon Product Classification
### Abstract
The purpose of this project is to create and analyze different neural network designs against one another. We use these designs in predicting and assigning categories for an Amazon product. We run an image recognition algorithm versus a tokenizing text categorization algorithm (testing multiple models with each) to see which is better for prediction. The data used in this project will be from a publicly available dataset (included in the references) that we parse, split, and clean to make the data useful for model training and testing. The data contains 10 categories that are assigned to products. After reaching the best possible prediction accuracies for both image recognition and text categorization, we ensemble our models to achieve a better accuracy. Read our paper here: []

### Running Notebooks
Models were trained with TensorFlow version 2.8.3 (images) and PyTorch version 2.2.1 (text). If there are issues with loading models, try downgrading to these versions.
To run the notebooks, our preprocessed dataset is required. Filepaths should be changed to the filepath of the data on your machine. Download our preprocessed data:

Images: To generate the image models for the CNN or CNN + Inception, run their respective .ipynb notebooks with a properly parsed dataset of images from running the image scripts on the amazon dataset.

Text Model: The step by step process for loading in or training the text models is in the output.ipynb jupyter notebook. The models that are available are:
1. A 1D CNN
2. An LSTM
3. A Transformer Encoder

Ensemble/Stacked Model: The second half of the output.ipnyb notebook goes through how to setup our various ensembling methods with all the previous base models, namely:
1. Weighted Averaging 
2. A meta-model trained on base model inputs.

(Note: The notebook provided works on the test_remove_filtered.csv file, which is a small sample csv of the real file. For the real parsed data, you are welcome to use our parsing functions provided in the GitHub or reach out to us for the preprocessed data. 

Sources:
Dataset: http://snap.stanford.edu/data/amazon/productGraph/
