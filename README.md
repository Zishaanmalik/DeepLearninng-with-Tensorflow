# Deep Learning with TensorFlow

This repository contains structured Jupyter Notebook implementations of core **deep learning architectures and training techniques** using **TensorFlow (Keras API)**.  
The focus is on understanding **model construction, training behavior, regularization, and architectural differences**, rather than building production pipelines.

The notebooks are organized as **independent experiments**, each demonstrating a specific concept.

---

## Covered Concepts

### 1. TensorFlow Fundamentals
- Tensor creation and manipulation
- Constants vs variables
- Basic TensorFlow operations and graph behavior

**Notebook**
- `Tensors_Consts_Variables.ipynb`

---

### 2. Model Construction APIs
Comparison of different TensorFlow model-building approaches:
- **Sequential API** — linear stack of layers
- **Functional API** — multi-input / multi-output architectures
- **Model Subclassing** — custom forward logic using `call()`

**Notebooks**
- `Sequential_method.ipynb`
- `Functional_method.ipynb`
- `Model-builder_method.ipynb`

---

### 3. Regression Models
- Dense neural networks for regression tasks
- Loss functions (MSE)
- Optimizer behavior
- Effect of dropout on regression stability

**Notebooks**
- `Regression_example.ipynb`
- `Dropouts_regression.ipynb`

---

### 4. Classification Models
- Softmax and sigmoid outputs
- Cross-entropy loss
- Dropout for overfitting control
- Accuracy vs loss behavior

**Notebooks**
- `Dropouts_classification.ipynb`
- `Digit_classification.ipynb`

---

### 5. Convolutional Neural Networks (CNN)
- Convolution layers
- Pooling operations
- Feature extraction for image data
- CNN training pipeline using TensorFlow

**Notebook**
- `CNN.ipynb`

---

### 6. Recurrent Neural Networks (RNN / LSTM)
- Sequence modeling using RNNs and LSTMs
- Integer encoding for sequence inputs
- Temporal dependency handling

**Notebooks**
- `LSTMs.ipynb`
- `RNN_integer_encoding.ipynb`

---

### 7. Regularization Techniques
- Dropout
- Batch Normalization
- Impact on convergence and generalization

**Notebook**
- `BatchNormalization.ipynb`

---

### 8. Hyperparameter Exploration
- Learning rate effects
- Epoch and batch size variation
- Basic tuning strategies (manual experimentation)

**Notebook**
- `Hyperparameter_Tuning.ipynb`

---

### 9. Transfer Learning
- Pretrained CNN usage
- Freezing and unfreezing layers
- Comparison with and without data augmentation

**Notebooks**
- `transfer-learning(data-augmentation).ipynb`
- `transfer-learning(without_data-augmentation).ipynb`

---

### 10. Model Weights & Datasets
- Inspecting and modifying model weights
- Loading and training on custom datasets

**Notebooks**
- `get_and_set_weights.ipynb`
- `costum_dataset.ipynb`

---

## Environment

- Python 3.13  
- TensorFlow (Keras API)  
- NumPy, Matplotlib  
- Jupyter Notebook  

Install dependencies:
```bash
pip install tensorflow numpy matplotlib jupyter
