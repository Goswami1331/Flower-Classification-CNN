# 🌷 Multiclass Flower Species Classification using CNN

## Table of Contents
1. Introduction
2. Problem Statement
3. Dataset Used
4. Model Architecture
5. Instructions for Running the Code
6. Evaluation Metrics and Results
7. Insights and Challenges

---

## 1. Introduction

This project implements a Convolutional Neural Network (CNN) using the **TensorFlow/Keras** framework for a **multiclass image classification** task. The objective is to classify images into one of five distinct flower species, showcasing the ability of CNNs to learn complex, fine-grained visual features necessary for accurate categorization.

## 2. Problem Statement

The goal is to develop an image classification model that accurately distinguishes between five species: **Daisy, Dandelion, Rose, Sunflower, and Tulip**. This task requires robust feature extraction to handle visual similarities and variations caused by lighting and angle.

## 3. Dataset Used

* **Classes:** 5 unique flower species.
* **Source:** A public domain subset of a flower classification dataset.
* **Preprocessing:** Images were normalized to the $[0, 1]$ range and dynamically augmented (rotation, shift, flip) during training via the `ImageDataGenerator` to expand the training data and improve generalization. The data is organized into class-specific folders within the `data/flower_dataset/` directory.

## 4. Model Architecture

* **Framework:** TensorFlow (Keras API)
* **Input Size:** $64 \times 64$ RGB image
* **Optimizer:** Adam
* **Loss Function:** Categorical Crossentropy
* **Network Layers:**
    * **2 Convolutional Blocks:** Using 32 and 64 filters, each followed by **MaxPooling2D** and **Dropout (0.25)** for regularization.
    * **Classification Head:** A **Flatten** layer feeds into a $\text{Dense}(\mathbf{128}, \text{relu})$ layer with $\text{Dropout}(\mathbf{0.5})$, ending with a $\text{Dense}(\mathbf{5}, \text{softmax})$ output layer.

---

## 5. Instructions for Running the Code

### Prerequisites
* Python 3.7+
* Dependencies: Listed in requirements.txt.

i. Clone the repository
```bash
git clone [INSERT YOUR GITHUB REPO URL HERE]  
cd Flower-Classification-Project
```

ii. Install All Dependencies

```bash

pip install -r requirements.txt
```

iii. Add Your Data Ensure the flower images are placed inside the data/flower_dataset/ folder, with one subfolder for each of the five classes.

iv. Train the Model

```bash

python run_train_flowers.py
```

v. Simulate Prediction (Requires run_predict.py to be in the root)

```bash

python .\run_predict.py test_rose.jpg
```




## 📊 Section 6: Results 


## 6. Evaluation Metrics and Results

The model was trained for 20 epochs. The final performance was measured on the 20% validation split of the dataset.

| Metric | Result |
| :--- | :--- |
| :--- | :--- |
| **Validation Loss** | **1.1821** |
| **Validation Accuracy** | **0.5385 (53.85%)** |



## 7. Insights and Challenges

The achieved accuracy of **53.85%** is significantly higher than the 20% random guess baseline, confirming the CNN successfully learned discriminative features. The primary challenges addressed were:
* **Overfitting:** Mitigated by aggressive use of **Dropout layers** and dynamic **Data Augmentation** to enhance the model's ability to generalize.
* **Multiclass Confusion:** The high loss (**1.1821**) suggested confusion between visually similar species, addressed by increasing the capacity of the final Dense layer to 128 neurons.


