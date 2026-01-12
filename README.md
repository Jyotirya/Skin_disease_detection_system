# Skin Disease Detection System

## Overview

The Skin Disease Detection System is an end-to-end deep learning project focused on automated classification of skin diseases from medical images. The project explores multiple modeling strategies, ranging from training a custom Convolutional Neural Network (CNN) from scratch to using transfer learning with pretrained models. It further incorporates model explainability and uncertainty estimation, which are essential for building trustworthy medical AI systems.

This repository is designed as a research-oriented and application-ready pipeline, emphasizing not only predictive performance but also interpretability, robustness, and reliability—key requirements for medical AI applications.

---

## Objectives

- Develop a deep learning–based system for skin disease image classification  
- Compare custom CNNs with pretrained transfer learning models  
- Improve performance under severe class imbalance  
- Apply explainability techniques to interpret model predictions  
- Estimate predictive uncertainty to assess model confidence  
- Build a reproducible and extensible medical imaging pipeline  

---

## Dataset

The dataset consists of labeled skin lesion images categorized into seven disease classes. Images are organized class-wise and preprocessed for CNN compatibility. The data is split into training and test sets.

The dataset is highly imbalanced, with one dominant class accounting for approximately 67% of the samples, while multiple minority classes represent less than 5% each. This motivates the use of weighted evaluation metrics and uncertainty-aware analysis.

---

## Notebook Description

### CNN.ipynb
Implements a custom convolutional neural network trained from scratch to establish a baseline and understand dataset complexity.

### pretrained.ipynb
Applies transfer learning using pretrained CNN backbones, freezing early layers and selectively fine-tuning higher layers to improve generalization.

### Explainability_Uncertainity.ipynb
Adds interpretability and reliability by visualizing salient lesion regions and estimating predictive uncertainty for ambiguous cases.

---

## Methodology

### Data Preprocessing

- Image resizing to a fixed resolution  
- Pixel value normalization  
- Data augmentation using random flips, rotations, and zooming  

These steps reduce overfitting and improve generalization.

---

### Baseline Model: Custom CNN

The baseline CNN consists of stacked convolutional layers with ReLU activation, max-pooling for spatial downsampling, and fully connected layers for classification, followed by a softmax output layer.

Training is performed using categorical cross-entropy loss with the Adam optimizer.

---

### Transfer Learning with Pretrained Models

Pretrained ImageNet-based CNN backbones are used as feature extractors. Early layers are frozen to retain generic visual representations, while selective fine-tuning of higher layers enables data-efficient learning on medical images.

---

## Model Evaluation and Results

### Class Distribution

The dataset exhibits strong class imbalance, with a single majority class dominating the distribution. As a result, weighted metrics and AUC are emphasized over raw accuracy.

---

### Training Set Performance

- AUC: 0.927  
- Accuracy: 0.663  
- Weighted Precision: 0.81  
- Weighted Recall: 0.763  
- Weighted F1-score: 0.78  

---

### Test Set Performance

- AUC: 0.915  
- Accuracy: 0.648  
- Weighted Precision: 0.791  
- Weighted Recall: 0.748  
- Weighted F1-score: 0.77  

The close alignment between training and test metrics indicates stable generalization with minimal overfitting. High AUC values demonstrate strong class separability despite dataset imbalance.

---

### Key Observations

- High AUC (>0.91) indicates robust discrimination capability  
- Weighted metrics significantly outperform macro averages due to imbalance  
- Minority classes show reduced precision, motivating uncertainty estimation  
- Transfer learning provides clear gains over training from scratch  

---

## Explainability and Uncertainty

### Explainability

Gradient-based attribution methods generate heatmaps highlighting lesion regions influencing predictions, confirming clinically meaningful feature learning.

### Uncertainty Estimation

Predictive uncertainty is incorporated to flag low-confidence predictions, improving robustness and reducing overconfident errors in medical decision-making.

---

## Technologies Used

- Python  
- PyTorch / TensorFlow  
- NumPy  
- Matplotlib  
- Jupyter Notebook  
- Pretrained CNN architectures  

---

## Conclusion

This project presents a complete deep learning pipeline for skin disease classification, progressing from baseline CNNs to transfer learning and finally to explainable and uncertainty-aware systems. By combining strong discriminative performance with interpretability and confidence estimation, the system demonstrates practical readiness for real-world medical AI applications.
