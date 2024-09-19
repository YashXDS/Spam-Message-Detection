# Spam Detection System

A machine learning-based project that classifies text messages as **Spam** or **Ham** (not spam) using the **Naive Bayes** algorithm. This project demonstrates skills in text preprocessing, feature extraction, model building, and deploying a simple web application with **Streamlit** for real-time message classification.

## Project Overview

The Spam Detection System leverages a dataset of SMS messages and applies the following steps:
- Text cleaning and preprocessing
- Feature extraction using **CountVectorizer**
- Building a **Naive Bayes** model for classification
- Evaluating the model's performance using accuracy, precision, recall, and F1-score
- Deploying the model in a web interface using **Streamlit**

## Libraries & Tools

- **Pandas**: For data loading and manipulation.
- **Scikit-learn**: For machine learning model building, feature extraction, and evaluation.
  - `CountVectorizer` for transforming text into numerical features.
  - `Multinomial Naive Bayes` for text classification.
- **Pickle**: For saving and loading the trained model and vectorizer.
- **Streamlit**: To create an easy-to-use web app for classifying messages.
- **XAMPP** (optional): Used in other projects for database hosting, but not required for this project.

## Dataset

The dataset used is **SMSSpamCollection**, which consists of SMS messages labeled as either "spam" or "ham." The dataset is loaded in a tab-separated format, with two columns:
- **label**: The target label (either 'spam' or 'ham').
- **message**: The SMS text message.

## Setup

1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/spam-detection-system.git
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Update the path to the dataset (SMSSpamCollection) in the script as needed:
   ```python
   data = pd.read_csv("path/to/dataset/SMSSpamCollection", sep='\t', header=None, names=['label', 'message'])
   ```
4. Train the Naive Bayes model and save it along with the vectorizer:
   ```bash
   python train_model.py
   ```
5. Run the Streamlit web application:
   ```bash
   streamlit run app.py
   ```

## Project Components

### 1. Data Preprocessing
Text Cleaning: Messages are cleaned by converting them to lowercase and removing punctuation and extra spaces.
Label Encoding: Labels are converted to binary values, where 'spam' is mapped to 1 and 'ham' is mapped to 0.

### 2. Feature Extraction
Bag of Words Model: The text messages are transformed into numerical feature vectors using CountVectorizer from Scikit-learn.

### 3. Model Building
Naive Bayes Classifier: The Multinomial Naive Bayes algorithm is used to classify the text messages. It is a common algorithm for text classification tasks due to its simplicity and effectiveness.

### 4. Model Evaluation
The trained model is evaluated using the following metrics:
- **Accuracy**: Overall correctness of the model.
- **Precision**: Proportion of predicted spam messages that are actually spam.
- **Recall**: Proportion of actual spam messages that were correctly predicted.
- **F1-Score**: Harmonic mean of precision and recall.

### Model performance metrics:

```plaintext
Accuracy: 98%
Precision: 0.96
Recall: 0.95
F1 Score: 0.95
```

### 5. Web Application with Streamlit
A Streamlit web app is provided to allow users to input custom SMS messages and classify them as "Spam" or "Ham." The app performs the following steps:

- Takes input from the user via a text box.
- Cleans the input using the same preprocessing steps as the training data.
- Vectorizes the input using the trained CountVectorizer.
- Predicts whether the message is "Spam" or "Ham" using the trained Naive Bayes model.

## App Interface

- **Input Text**: Users can enter an SMS message.
- **Predict Button**: Upon clicking the button, the model makes a prediction.
- **Result Display: The app displays whether the message is classified as Spam or Ham.

## How It Works

- **Data Loading**: The dataset is loaded and preprocessed to make it ready for machine learning.
- **Model Training**: A Multinomial Naive Bayes model is trained on the text data.
- **Model Evaluation**: The model is evaluated on test data using various metrics.
- **Model Saving**: The trained model and vectorizer are saved using Pickle for later use in the web app.
- **Web App Deployment**: The Streamlit app uses the saved model and vectorizer to classify new user-provided SMS messages.

## Running the Application

### Model Training

- **Load the dataset**:

```python

data = pd.read_csv('SMSSpamCollection', sep='\t', header=None, names=['label', 'message'])
```

- **Clean and preprocess the text**:

```python

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text

data['message'] = data['message'].apply(clean_text)
```

- **Train the Naive Bayes classifier**:

```python

model = MultinomialNB()
model.fit(X_train, y_train)
```

- **Evaluate and save the model**:

```python
with open('spam_classifier_model.pkl', 'wb') as file:
    pickle.dump(model, file)
```

## Web App

- **Run the Streamlit app**:

```bash
streamlit run app.py
```

- **Enter an SMS message to classify it as spam or ham**:

```plaintext

Enter the message text:
[Predict]
The result is displayed:
```
```plaintext

The message is classified as: Spam / Ham
```

## Future Enhancements
- **Advanced NLP Techniques**: Integrate techniques like TF-IDF or Word Embeddings to improve model performance.
- **Real-Time Message Classification**: Extend the web app to classify messages in real-time with more advanced machine learning models.
- **Model Comparisons**: Compare performance with other machine learning algorithms like SVM or Random Forest.
- **Database Integration**: Use XAMPP or a similar service to store classified messages in a database.
