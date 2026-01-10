# Skin Disease Detection System

## Overview

The **Skin Disease Detection System** is an end-to-end deep learning project focused on automated classification of skin diseases from medical images. The project explores multiple modeling strategies, ranging from training a custom Convolutional Neural Network (CNN) from scratch to using transfer learning with pretrained models. It further incorporates model explainability and uncertainty estimation, which are essential for building trustworthy medical AI systems.

This repository is designed as a research-oriented and application-ready pipeline, emphasizing not only performance but also interpretability and reliability.

---

## Objectives

- Develop a deep learning–based system for skin disease image classification  
- Compare custom CNNs with pretrained transfer learning models  
- Improve classification performance using pretrained backbones  
- Apply explainability techniques to interpret model predictions  
- Estimate predictive uncertainty to assess model confidence  
- Build a reproducible and extensible medical imaging pipeline  

---

## Dataset

- The dataset consists of labeled skin lesion images categorized by disease type  
- Images are organized class-wise and preprocessed for CNN compatibility  
- Data is split into:
  - Training set  
  - Validation set  
  - Test set  

This is treated as a multi-class image classification problem where each image belongs to exactly one disease category.

---


## Notebook Description

### CNN.ipynb
- Implements a custom convolutional neural network trained from scratch  
- Establishes baseline performance  
- Helps understand dataset complexity  

### pretrained.ipynb
- Uses transfer learning with pretrained CNN architectures  
- Freezes early layers and fine-tunes higher layers  
- Improves accuracy and convergence speed  

### Explainability_Uncertainity.ipynb
- Applies explainability techniques to visualize model attention  
- Estimates uncertainty in model predictions  
- Enhances reliability and interpretability  

---

## Methodology

### 1. Data Preprocessing

- Image resizing to a fixed resolution  
- Pixel value normalization  
- Data augmentation techniques:
  - Random flips  
  - Rotations  
  - Zooming  

These steps improve generalization and reduce overfitting.

---

### 2. Baseline Model: Custom CNN

- Convolutional layers with ReLU activation  
- Max-pooling layers for spatial downsampling  
- Fully connected layers for classification  
- Softmax output layer for multi-class prediction  

**Training Configuration**
- Loss function: Categorical Cross-Entropy  
- Optimizer: Adam  
- Metric: Accuracy  
- Epoch-based training with validation monitoring  

---

### 3. Transfer Learning with Pretrained Models

- Pretrained backbones (ImageNet-trained) are used  
- Early layers are frozen to retain learned features  
- Custom classification head is added  
- Optional fine-tuning with a lower learning rate  

**Advantages**
- Faster convergence  
- Better performance on limited data  
- Stronger feature extraction  

---

### 4. Model Evaluation

Models are evaluated using:
- Training vs validation accuracy curves  
- Validation loss trends  
- Final test set metrics  

Performance comparison is done between:
- Custom CNN  
- Pretrained transfer learning models  

---

## Explainability and Uncertainty

### Explainability

- Gradient-based visualization techniques are applied  
- Heatmaps highlight regions influencing predictions  
- Helps verify medically relevant feature learning  

### Uncertainty Estimation

- Quantifies prediction confidence  
- Identifies ambiguous or low-confidence samples  
- Reduces risk of overconfident incorrect predictions  

---

## Key Results and Insights

- Transfer learning outperforms training from scratch  
- Pretrained models generalize better  
- Explainability confirms focus on lesion regions  
- Uncertainty estimation improves decision safety  
- Balanced trade-off between accuracy and interpretability  

---

## Technologies Used

- Python  
- PyTorch / TensorFlow  
- NumPy  
- Matplotlib  
- Jupyter Notebook  
- Pretrained CNN architectures  

---
