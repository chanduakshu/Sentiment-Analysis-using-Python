# Sentiment Analysis using Machine Learning
# Algorithm: Logistic Regression
# Level: Beginner

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

# Sample training data
sentences = [
    "I love this product",
    "This is an amazing experience",
    "I am very happy",
    "I hate this",
    "This is bad",
    "I am disappointed"
]

labels = [1, 1, 1, 0, 0, 0]  # 1 = Positive, 0 = Negative

# Convert text to numeric data
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(sentences)

# Train model
model = LogisticRegression()
model.fit(X, labels)

# User input
print("Sentiment Analysis System")
text = input("Enter a sentence: ")

# Predict sentiment
text_vector = vectorizer.transform([text])
prediction = model.predict(text_vector)

if prediction[0] == 1:
    print("Sentiment: 😊 Positive")
else:
    print("Sentiment: 😞 Negative")

