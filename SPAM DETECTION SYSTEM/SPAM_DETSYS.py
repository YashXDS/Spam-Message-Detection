import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import re
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle

# Load dataset
data = pd.read_csv("SMSSpamCollection", sep='\t', header=None, names=['label', 'message'])

# Show the first few rows
print(data.head())

# Clean the text (lowercasing and removing punctuation)
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)  # Remove non-word characters
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    return text

data['message'] = data['message'].apply(clean_text)

# Convert labels to binary (spam=1, ham=0)
data['label'] = data['label'].map({'ham': 0, 'spam': 1})

# Feature extraction
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(data['message'])
y = data['label']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Naive Bayes model
model = MultinomialNB()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy:.2f}%') 
print('Classification Report:') 
print(report)


### **4. Model Evaluation**
# Compute metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f'Accuracy: {accuracy:.2f}')
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1 Score: {f1:.2f}')

# Save the model to disk
model_filename = 'spam_classifier_model.pkl'
with open(model_filename, 'wb') as file:
    pickle.dump(model, file)

# Save the vectorizer to disk
vectorizer_filename = 'vectorizer.pkl'
with open(vectorizer_filename, 'wb') as file:
    pickle.dump(vectorizer, file)

print("Model and vectorizer saved successfully.")


