# 🧠 Digit Recognizer with Neural Network

Welcome to the **Digit Recognizer** project! 🎯 This repository contains an implementation of a simple neural network built from scratch using **NumPy** and **Pandas** to recognize handwritten digits from the **Kaggle Digit Recognizer** dataset. 🚀

## 📌 Table of Contents
- [🧠 Digit Recognizer with Neural Network](#-digit-recognizer-with-neural-network)
  - [📌 Table of Contents](#-table-of-contents)
  - [📂 Dataset](#-dataset)
  - [⚙️ Installation](#️-installation)
  - [🛠️ Project Structure](#️-project-structure)
  - [📊 Data Preprocessing](#-data-preprocessing)
  - [🧠 Neural Network Architecture](#-neural-network-architecture)
    - [Activation Functions](#activation-functions)
  - [🚀 Training the Model](#-training-the-model)
  - [📝 Making Predictions](#-making-predictions)
  - [📤 Submitting to Kaggle](#-submitting-to-kaggle)
  - [📜 License](#-license)
    - [🎯 Happy Coding! 🚀](#-happy-coding-)

---

## 📂 Dataset
The dataset used in this project is the **Kaggle Digit Recognizer** dataset, which consists of images of handwritten digits (0-9) stored as **28x28 grayscale pixel values**.

- **Training Data:** `train.csv` (Contains labeled images)
- **Test Data:** `test.csv` (Contains unlabeled images for submission)

## ⚙️ Installation
To run this project, you need to install the following dependencies:
```bash
pip install numpy pandas matplotlib
```
Alternatively, if you're running this on Kaggle, the dependencies are already pre-installed. ✅

## 🛠️ Project Structure
```
├── digit_recognizer.py  # Main script
├── README.md            # Project documentation
├── submission.csv       # Final Kaggle submission file
```

## 📊 Data Preprocessing
1. **Load the dataset**: The training data is read using `pandas.read_csv()`.
2. **Shuffle the dataset**: Ensures randomness in training.
3. **Split into training & validation**: 1000 samples for validation, the rest for training.
4. **Normalize pixel values**: Scales pixel values from `[0, 255]` to `[0, 1]`.

## 🧠 Neural Network Architecture
The neural network consists of:
- **Input layer**: 784 neurons (28x28 pixels)
- **Hidden layer**: 10 neurons, activated using **ReLU**
- **Output layer**: 10 neurons (digits 0-9), activated using **Softmax**

### Activation Functions
- **ReLU (Rectified Linear Unit)**: `max(0, Z)`
- **Softmax**: Converts logits into probabilities.

## 🚀 Training the Model
The network is trained using **gradient descent** with the following steps:
1. **Forward propagation**: Computes activations for hidden and output layers.
2. **Backward propagation**: Calculates gradients using the loss function.
3. **Parameter updates**: Uses **learning rate (alpha = 0.1)** for optimization.

```python
W1, b1, W2, b2 = gradient_descent(X_train, Y_train, alpha=0.10, iterations=500)
```
The accuracy is printed every 10 iterations. 📈

## 📝 Making Predictions
The model predicts digits using the trained weights:
```python
def make_predictions(X, W1, b1, W2, b2):
    _, _, _, A2 = forward_prop(W1, b1, W2, b2, X)
    predictions = get_predictions(A2)
    return predictions
```
Example test prediction:
```python
test_prediction(0, W1, b1, W2, b2)
```

## 📤 Submitting to Kaggle
After training, predictions are made on the test set and saved in `submission.csv`.
```python
submission = pd.DataFrame({
    'ImageId': range(1, len(predictions) + 1),
    'Label': predictions
})
submission.to_csv('submission.csv', index=False)
```
Now, upload `submission.csv` to **Kaggle** to see your leaderboard score! 🏆

## 📜 License
This project is **MIT licensed**. Feel free to use and modify it. 💡

---
### 🎯 Happy Coding! 🚀

