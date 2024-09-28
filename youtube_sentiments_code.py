import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline

# Load dataset
data = pd.read_csv("C:/Users/Abhinav Bhardwaj/Videos/My apps/tra.csv")

# Handle missing values
data['Comment'].fillna("", inplace=True)

# Assign labels based on polarity column
data['label'] = data['polarity'].apply(lambda x: 'positive' if x == 'pos' else 'negative')

# Count the number of positive and negative comments
positive_count = (data['label'] == 'positive').sum()
negative_count = (data['label'] == 'negative').sum()

# Plot the bar graph
plt.figure(figsize=(8, 6))
plt.bar(['Positive', 'Negative'], [positive_count, negative_count], color=['green', 'red'])
plt.title('Number of Positive and Negative Comments')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.show()

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(data['Comment'], data['label'], test_size=0.2, random_state=42)

# Build pipeline for text classification with SVM
text_clf = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', SVC(kernel='linear')),  # Linear kernel is often used for text classification
])

# Train the model
text_clf.fit(X_train, y_train)

# Predict sentiment on test set
predicted = text_clf.predict(X_test)

# Evaluate accuracy
accuracy = accuracy_score(y_test, predicted)
accuracy_percentage = accuracy * 100
print("Accuracy:", accuracy_percentage, "%")

# Classification report for test set
print("Classification Report for Test Set:")
print(classification_report(y_test, predicted))

# Classification report for training set
train_predicted = text_clf.predict(X_train)
print("Classification Report for Training Set:")
print(classification_report(y_train, train_predicted))

# Additional Metrics
precision = precision_score(y_test, predicted, average='weighted')
recall = recall_score(y_test, predicted, average='weighted')
f1 = f1_score(y_test, predicted, average='weighted')

print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
