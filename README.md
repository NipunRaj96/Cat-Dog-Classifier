# Cat 🐈 vs Dog 🐶 Image Classification Project 

---

**Overview**
- This project implements a deep learning model to classify images as either cats or dogs using a pre-trained ResNet50 model with TensorFlow/Keras achieving an accuracy of **`98.76%`** on both training and testing data.
- The dataset is sourced from Kaggle's "Dogs vs. Cats" dataset, and the model is trained with data augmentation to improve generalization and reduce overfitting.

---

**Dataset**
- **Source**: Kaggle dataset: [`salader/dogs-vs-cats`](https://www.kaggle.com/datasets/salader/dogs-vs-cats)
- **Structure**:
  - **Training Set**: 20,000 images (10,000 cats, 10,000 dogs).
  - **Test Set**: 5,000 images (2,500 cats, 2,500 dogs).
- **Path**: Dataset files are located at `/kaggle/input/dogs-vs-cats`, with subdirectories `train` and `test` containing `cats` and `dogs` folders.

---

**Prerequisites**
- Python 3.11
- TensorFlow/Keras
- `kagglehub` for dataset download
- NumPy
- A valid Kaggle API token (`kaggle.json`) placed in the `~/.kaggle/` directory

---

**Model Architecture**
- **Base Model**: Pre-trained ResNet50 (weights from ImageNet, excluding top layers).
- **Custom Layers**:
  - Global Average Pooling
  - Dense layer (512 units, ReLU activation)
  - Dropout (0.5)
  - Output layer (1 unit, sigmoid activation for binary classification)
- **Total Parameters**: 24,637,313 (23,587,712 non-trainable, 1,049,601 trainable).
- **Input Shape**: Images resized to 224x224 pixels with 3 color channels (RGB).

---

**Data Preprocessing**
- **Data Augmentation (Training)**:
  - Rotation range: 20 degrees
  - Width/height shift: 20%
  - Shear range: 20%
  - Zoom range: 20%
  - Horizontal flip: Enabled
  - Fill mode: Nearest
- **Preprocessing**: Applied `preprocess_input` from ResNet50 to normalize pixel values.
- **Validation**: Images are checked for validity using PIL to ensure non-corrupted files with non-zero size.

---

**Training**
- **Batch Size**: 32
- **Epochs**: 5 (with early stopping based on loss, patience=5)
- **Optimizer**: Adam (learning rate=1e-4)
- **Loss Function**: Binary cross-entropy
- **Metrics**: Accuracy
- **Callbacks**:
  - EarlyStopping: Monitors loss, restores best weights after 5 epochs without improvement.
  - ModelCheckpoint: Saves the best model based on accuracy to `cat_dog_classifier.h5`.
- **Note**: Training may take significant time due to the large dataset and model complexity. To reduce training time, consider using a smaller batch size, fewer epochs, or a GPU/TPU for acceleration.

---

**Data Loading**
- **Training Data**: Loaded using `ImageDataGeneratorunder_scoreflow_from_directory` from `train_dir`.
- **Test Data**: Loaded similarly from `test_dir` without shuffling.
- **Classes**: Binary classification (cats=0, dogs=1).

---

**Model Summary**

The model summary details the ResNet50 architecture followed by custom layers. Key layers include:
- **Input Layer**: 224x224x3
- **ResNet50 Layers**: Multiple convolutional, batch normalization, and ReLU activation layers.
- **Custom Layers**: GlobalAveragePooling2D, Dense(512, ReLU), Dropout(0.5), Dense(1, sigmoid).

---

**Notes**
- (`selfunder_scorewarn_if_super_not_filled()`), which does not affect functionality.
- Ensure the dataset directories (`train_dir`, `test_dir`) and image files are accessible to avoid data loading errors.

---

**Future Improvements**
- Implement fine-tuning of ResNet50 layers (currently set to non-trainable).
- Increase epochs or adjust learning rate for better convergence.
- Add validation split to monitor performance during training.
- Explore additional data augmentation techniques or other pre-trained models.
