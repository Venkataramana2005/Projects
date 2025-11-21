# Kidney CT Image Classification

This project focuses on classifying Kidney CT images using deep learning models trained on the RadImageNet dataset along with custom architectures.

### Purpose

The goal is to develop accurate and robust models that can classify kidney CT images into their respective categories, improving medical imaging analysis using deep learning.

### Methodology

Uses TensorFlow/Keras for model development and training.

### Implements three models:

 - ResNet50 (RadImageNet pretrained)
 - DenseNet121 (RadImageNet pretrained)
 - Simple Custom CNN

### Includes complete workflow:

- Data loading and preprocessing
- Data augmentation
- Model building and training
- Evaluation with accuracy, loss curves, confusion matrix, and classification report
- Compares model performance across all architectures.

### Requirements

 - Python (compatible version)
 - TensorFlow / Keras
 - NumPy
 - Matplotlib
 - Seaborn
 - scikit-learn
 - pandas
 - Jupyter Notebook
 - (Optional) RadImageNet model weights

### Usage

1. Install required dependencies using:
 'pip install -r requirements.txt'

2. Open and run the notebook:
'kidney_ct_classification.ipynb'

3.Follow the notebook cells for:

- Dataset preparation
- Model selection (ResNet50 / DenseNet121 / Simple CNN)
- Training and evaluation
- Visualizing metrics and generating results

### Outputs

- Training and validation accuracy/loss plots
- Confusion matrices
- Classification reports
- Saved trained model files (.h5)
