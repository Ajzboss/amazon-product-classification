# Investigating Deep Learning Models for Amazon Product Classification
### Abstract
The purpose of this project is to create and analyze different neural network designs against one another. We use these designs in predicting and assigning categories for an Amazon product. We run an image recognition algorithm versus a tokenizing text categorization algorithm (testing multiple models with each) to see which is better for prediction. The data used in this project will be from a publicly available dataset (included in the references) that we parse, split, and clean to make the data useful for model training and testing. The data contains 10 categories that are assigned to products. After reaching the best possible prediction accuracies for both image recognition and text categorization, we ensemble our models to achieve a better accuracy. Read our paper here: []

### Running Notebooks
Models were trained with TensorFlow version 2.8.3 (images) and PyTorch version 2.2.1 (text). If there are issues with loading models, try downgrading to these versions.
To run the notebooks, our preprocessed dataset is required. Filepaths should be changed to the filepath of the data on your machine. Download our preprocessed data:

Images: []

Text: []

Sources:
Dataset: http://snap.stanford.edu/data/amazon/productGraph/
