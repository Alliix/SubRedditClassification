import streamlit as st
import pickle
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
import pandas as pd

stop_words = set(stopwords.words('english'))
ps = nltk.PorterStemmer()

def CleanPost(post):
#      RegEx
    zeroSpaceWidth = re.compile(r'&#x200B')
    urls = re.compile(r'(https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]+\.[^\s]{2,}|www\.[a-zA-Z0-9]+\.[^\s]{2,})', re.IGNORECASE)
    numbers = re.compile(r' \d+(\.\d+)? ')
    punctuation = re.compile(r'[^\w\d\s]')
    whitespaces = re.compile(r'\s+')
    leadTrailWhitespace = re.compile(r'^\s+|\s+?$')

    x = post
    # Replace zero width space with ' '
    x = zeroSpaceWidth.sub(' ',x)
    # Replace URLs with 'url'
    x = urls.sub('url',x)
    # Replace numbers with 'nmbr'
    x = numbers.sub('nmbr',x)
    # Remove punctuation
    x = punctuation.sub(' ',x)
    # Replace whitespace between terms with ' '
    x = whitespaces.sub(' ',x)
    # Remove leading and trailing whitespace
    x = leadTrailWhitespace.sub(' ',x)

    text_tokens = word_tokenize(x)
    # Remove word stems using a Porter stemmer
    tokens_without_ws = [ps.stem(word) for word in text_tokens]
    x = (" ").join(tokens_without_ws)

    # remove stop words from posts
    text_tokens = word_tokenize(x)
    tokens_without_sw = [word for word in text_tokens if not word in stop_words]
    x = (" ").join(tokens_without_sw)
    
    return x

def getWordFrequency(processedPostOutputFile):
    posts = pd.read_csv(processedPostOutputFile)

    allPostsConcat = ''
    for post in posts.Post:
        if(type(post)==str):
            allPostsConcat+=post

    # create bag-of-words
    all_words = []

    words = word_tokenize(allPostsConcat)
    for word in words:
        all_words.append(word)

    all_words = nltk.FreqDist(all_words)
    
    return all_words

all_words_combined = getWordFrequency('all_posts_processed.csv')
word_features = list(all_words_combined.keys())[:1500]

def find_features(post):
    words = word_tokenize(post)
    features = {}
    for word in word_features:
        features[word] = (word in words)
    return features

@st.cache(allow_output_mutation=True)
def load_subreddit_model():
    return pickle.load(open('finalized_model_subreddits.sav', 'rb'))

@st.cache(allow_output_mutation=True)
def load_depressed_model():
    return pickle.load(open('finalized_model_depression_or_not.sav', 'rb'))

def ClassifyPost(post):
    post = CleanPost(post)
    subreddit_model = load_subreddit_model()
    depression_model = load_depressed_model()
    
    featureset = find_features(post)
    
    result = subreddit_model.classify(featureset)
    return result

def IsDepressionPost(post):
    post = CleanPost(post)
    depression_model = load_depressed_model()
    
    featureset = find_features(post)
    
    result = depression_model.classify(featureset)
    return result

def show_predict_page():
    st.title("Reddit Post Classifier")

    post = st.text_area("Enter Post")

    ok = st.button("Classify")
    if ok:
        classifyResult = ClassifyPost(post)
        # depressionResult = IsDepressionPost(post)

        st.subheader('Post is likely from r/'+classifyResult)
        # st.subheader('Post is likely '+depressionResult)