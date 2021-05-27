import streamlit as st
import pickle
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
nltk.download('webtext')
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn import model_selection



def find_features_LM(post):
    words = word_tokenize(post)
    features = {}
    
    with open('./WordNetLemmatizer/word_features.data', 'rb') as filehandle:
        # read the data as binary data stream
        word_features = pickle.load(filehandle)
    
    for word in word_features:
        features[word] = (word in words)
    return features

def show_explore_page_LM():
    st.title("Explore model Using WordNetLemmatizer")
    loaded_model = pickle.load(open('./WordNetLemmatizer/Models/finalized_model_subreddits.sav', 'rb'))
    
    allPosts = pd.read_csv('./WordNetLemmatizer/all_posts_processed.csv')
    posts_all = list(zip(allPosts.loc[:,"Post_Parsed"].values,allPosts.loc[:,"Subreddit_Code"].values))
    featuresets = [(find_features_LM(str(text)), label) for (text, label) in posts_all]
    seed = 1
    training, testing = model_selection.train_test_split(featuresets, test_size = 0.25, random_state=seed)
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
    # 
    # 

    loaded_model = pickle.load(open('./WordNetLemmatizer/Models/finalized_model_is_depression.sav', 'rb'))
    
    allPosts = pd.read_csv('./WordNetLemmatizer/all_posts_processed.csv')
    posts_all = list(zip(allPosts.loc[:,"Post_Parsed"].values,allPosts.loc[:,"Is_Depression_Code"].values))
    featuresets = [(find_features_LM(str(text)), label) for (text, label) in posts_all]
    seed = 1
    training, testing = model_selection.train_test_split(featuresets, test_size = 0.25, random_state=seed)
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