# Sentiment Analysis Using LSTM on IMDB Movie Reviews

## Project Description
This project implements a sentiment analysis model using Long Short-Term Memory (LSTM) networks to classify IMDB movie reviews as either positive or negative. The project demonstrates the end-to-end pipeline, from data preprocessing to model evaluation, for a binary classification task. The insights derived from this model can be applied to various business use cases, including product review analysis, customer feedback monitoring, and marketing strategies.

---

## Business Use Cases
- **Product Reviews Analysis**: Analyze customer sentiment on products to improve offerings.
- **Customer Feedback Monitoring**: Identify trends in customer satisfaction.
- **Marketing and Advertising**: Tailor campaigns based on audience sentiment.

---

## Project Workflow

### 1. Dataset Loading
- Use TensorFlow's built-in IMDB dataset (`tf.keras.datasets.imdb`).
- Load 25,000 reviews for training and 25,000 for testing, labeled as positive or negative.
- Inspect the dataset structure, which consists of integer-encoded words and corresponding labels.

### 2. Data Exploration
- Visualize:
  - Distribution of review lengths.
  - Label counts (positive vs. negative).
- Analyze vocabulary size and encoding to better understand the dataset.

### 3. Text Preprocessing
- **Padding**: Use `pad_sequences` to ensure uniform input size for the LSTM model.
- **Decoding**: Optionally decode integer-encoded reviews back to text for interpretability.

### 4. Build the LSTM Model
- Use an **Embedding Layer** to map integer-encoded words into dense vector representations.
- Add one or more **LSTM Layers** to capture sequential dependencies in the text data.
- Compile the model using an appropriate optimizer and loss function.

### 5. Train the Model
- Use the **binary cross-entropy** loss function and appropriate metrics.
- Train on the training dataset and validate using the test dataset.

### 6. Evaluate the Model
- Calculate:
  - **Accuracy**
  - **Precision**
  - **Recall**
  - **F1-Score**
- Plot learning curves for loss and accuracy during training and validation.

### 7. Make Predictions
- Test the model with custom reviews.
- Decode predictions into "positive" or "negative" sentiments for interpretability.

---

## Results
The LSTM model is evaluated based on the following metrics:
- **Accuracy**
- **F1-Score**
- **Precision**
- **Recall**
- **Binary Cross-Entropy Loss**

The model achieves satisfactory performance with minimum loss, demonstrating its effectiveness for sentiment analysis tasks.

---

## Technical Tags
- **Natural Language Processing (NLP)**
- **LSTM**
- **Classification**
- **TensorFlow/Keras**

---