import streamlit as st
import pickle
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
nltk.download('webtext')
from sklearn.metrics import classification_report, confusion_matrix


def find_features_LM(post):
    words = word_tokenize(post)
    features = {}
    
    with open('./../WordNetLemmatizer/word_features.data', 'rb') as filehandle:
        # read the data as binary data stream
        word_features = pickle.load(filehandle)
    
    for word in word_features:
        features[word] = (word in words)
    return features

def show_explore_page_LM():

# 
# Subreddit Classifier
# 

    st.title("Explore model Using WordNetLemmatizer")
    loaded_model = pickle.load(open('./../WordNetLemmatizer/Models/finalized_model_subreddits.sav', 'rb'))
    
    # read saved testing data
    with open('./../WordNetLemmatizer/Models/testing_subreddits.data', 'rb') as filehandle:
        testing = pickle.load(filehandle)

    txt_features, labels = list(zip(*testing))
    
    accuracy = nltk.classify.accuracy(loaded_model, testing)*100
    st.subheader("Subreddit Voting Classifier: Accuracy: {}".format(accuracy))

    prediction = loaded_model.classify_many(txt_features)
    report = classification_report(labels, prediction)
    st.write(report)

    cm = pd.DataFrame(
    confusion_matrix(labels, prediction),
    index = [['actual', 'actual','actual', 'actual'], ['depression', 'unpopularopinion', 'lonely', 'MachineLearning']],
    columns = [['predicted', 'predicted','predicted', 'predicted'], ['depression', 'unpopularopinion', 'lonely', 'MachineLearning']])
    st.dataframe(cm)

    # 
    # Is r/depression Classifier
    # 

    loaded_model = pickle.load(open('./../WordNetLemmatizer/Models/finalized_model_is_depression.sav', 'rb'))
    
    # read saved testing data
    with open('./../WordNetLemmatizer/Models/testing_depression.data', 'rb') as filehandle:
        testing = pickle.load(filehandle)

    txt_features, labels = list(zip(*testing))

    accuracy = nltk.classify.accuracy(loaded_model, testing)*100
    st.subheader("r/depression Voting Classifier: Accuracy: {}".format(accuracy))
    
    prediction = loaded_model.classify_many(txt_features)
    report = classification_report(labels, prediction)
    st.write(report)

    cm = pd.DataFrame(
    confusion_matrix(labels, prediction),
    index = [['actual', 'actual'], ['r/depression', 'not r/depression']],
    columns = [['predicted', 'predicted'], ['depression', 'not r/depression']])
    st.dataframe(cm)