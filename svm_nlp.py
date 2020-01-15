'''
In previous NLP iterations we used a NaiveBayesAnalyzer for sentiment analysis
that had been trained on movie review data. In hopes to improve accuracy, we are
attempting to instead use a model trained on our own data, which we do here with
an SVM classifier.

This script reads a csv file with one column of raw data (verbatim customer
responses) and one column of manual classifications (0 for negative, 1 for
positive). Specify which column contains which under "Constants and Variables".
'''

import numpy as np
import pandas as pd
import time
import re
import nltk
from nltk.corpus import stopwords
#from spellchecker import SpellChecker
from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from joblib import dump

################################################################################
## CONSTANTS AND VARIABLES
################################################################################

T_0 = time.time()

# Upload a csv with one column raw responses, one column manually decided categories
INPUTFILE = r"C:\Users\jenni\Desktop\imdb.csv"

# Name of output file comparingpredictions to actual values
OUTPUTFILE = r"C:\Users\jenni\Desktop\nlp_output.csv"

ANALYZERFILE = "SVM_sentiment_analyzer.sav"

TRAIN_SPLIT = 0.85 # proportion of records to use for training

SPACE = ' '

TEST = True # Toggle testing on data loaded here

# Import dataset (local CSV for now), specify responses, output class labels
responses = pd.read_csv(INPUTFILE, encoding='latin-1')
features = responses.iloc[:, 0]
labels = responses.iloc[:, 1]

#spell = SpellChecker() # spellchecker

# TF-IDF vectorizer
vectorizer = TfidfVectorizer(max_features=2000, min_df=4, max_df=0.9, stop_words=stopwords.words('english'))

text_classifier = svm.SVC(kernel='linear')

################################################################################
## MAIN CALLS
################################################################################

# create a list of prepared responses for analysis
processed_features = []
# clean up text but don't lemmatize and return to single string
for response in range(len(features)):
    # Remove all the special characters
    processed_feature = re.sub(r'\W', ' ', str(features[response]))
    # Remove all single characters
    processed_feature= re.sub(r'\s+[a-zA-Z]\s+', ' ', processed_feature)
    # Remove single characters from the start
    processed_feature = re.sub(r'\^[a-zA-Z]\s+', ' ', processed_feature)
    # Substituting multiple spaces with single space
    processed_feature = re.sub(r'\s+', ' ', processed_feature, flags=re.I)
    # Removing prefixed 'b'
    processed_feature = re.sub(r'^b\s+', '', processed_feature)
    # Converting to Lowercase
    processed_feature = processed_feature.lower()
    # Correct spelling and lemmatize (broken into list)
    words = processed_feature.split()
    #for i in range(len(words)):
    #    if words[i] in spell.unknown(words):
    #        words[i] = spell.correction(words[i])
    # Return each response to a single string
    processed_feature = SPACE.join(words)

    processed_features.append(processed_feature)
print("Text preparation completed.")


# Vectorize text holding on to top TF-IDF scoring words
vectorized_features = vectorizer.fit_transform(processed_features).toarray()
print("TF-IDF completed.")

if TEST==True:
    # Split data into 80% for training, 20% for testing
    X_train, X_test, y_train, y_test = train_test_split(vectorized_features, labels, test_size=(1-TRAIN_SPLIT), random_state=0)
    # X_train, y_train = vectorized_features[0:int(TRAIN_SPLIT*len(vectorized_features))], labels.head(int(TRAIN_SPLIT*len(vectorized_features)))
    # X_test, y_test = vectorized_features[len(X_train):len(vectorized_features)], labels.tail(len(labels) - len(X_train))
if TEST==False:
    X_train, y_train = vectorized_features, labels

# Train the model
text_classifier.fit(X_train, y_train)

# Save the trained model to a file
dump(text_classifier, ANALYZERFILE)

if TEST==True:
    # Use model on testing data
    predictions = text_classifier.predict(X_test)

    # Output metrics assessing accuracy of model
    print(confusion_matrix(y_test, predictions))
    print(classification_report(y_test, predictions))
    print("Accuracy: " +  str(accuracy_score(y_test, predictions)))

    test_features = features.tail(len(X_test)).reset_index()
    test_processed_features = processed_features[len(X_train):len(processed_features)]
    test_output = pd.concat([test_features, pd.Series(test_processed_features), y_test.reset_index().iloc[:, 1], pd.Series(predictions)], axis=1, ignore_index=True)
    test_output.columns = ("orig_index", "Verbatim", "Processed Feature", "Manual Sentiment", "Model Prediction")
    test_output.to_csv(OUTPUTFILE, index=False)

runtime = time.time() - T_0
print("Runtime: " + str(runtime))
