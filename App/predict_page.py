import streamlit as st
import pickle
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# 
# NO NORMALIZATION
# 

def cleanPost(post):
    urls = re.compile(r'(https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]+\.[^\s]{2,}|www\.[a-zA-Z0-9]+\.[^\s]{2,})')
    numbers = re.compile(r'\d+(\.\d+)?')
    posessivePronouns = re.compile(r"’s")
    apostrophe=re.compile(r"’")
    someSigns =re.compile(r"\\n|\\r")
    punctuation = re.compile(r"[^\w\s]")
    whitespaces = re.compile(r'\s+')
    leadTrailWhitespace = re.compile(r'^\s+|\s+?$')
    
    cleanPost = post.lower()
    cleanPost = urls.sub('url',cleanPost)
    cleanPost = numbers.sub('nmbr',cleanPost)
    cleanPost = posessivePronouns.sub('',cleanPost)
    cleanPost = apostrophe.sub('',cleanPost)
    cleanPost = someSigns.sub('',cleanPost)
    cleanPost = punctuation.sub(' ',cleanPost)
    cleanPost = whitespaces.sub(' ',cleanPost)
    cleanPost = leadTrailWhitespace.sub('',cleanPost)
    
    # stop words
    with open('./../stop_words_no_punct.data', 'rb') as filehandle:
        # read the data as binary data stream
        stop_words_no_punct = pickle.load(filehandle)
        
    post_without_stop_words = []

    text_tokens = word_tokenize(cleanPost)
    tokens_without_stop_words = [word for word in text_tokens if not word in stop_words_no_punct]
    post_without_stop_words = (" ").join(tokens_without_stop_words)

    cleanPost = post_without_stop_words
    
    return cleanPost

def find_features_NN(post):
    words = word_tokenize(post)
    features = {}
    
    #read generated features for Lemmatizer
    with open('./../NoNormalization/word_features.data', 'rb') as filehandle:
        # read the data as binary data stream
        word_features = pickle.load(filehandle)
    
    for word in word_features:
        features[word] = (word in words)
    return features

def classifySubreddit_NN(post):
    modelFile = './../NoNormalization/Models/finalized_model_subreddits.sav'
    post = cleanPost(post)
    
    loaded_model = pickle.load(open(modelFile, 'rb'))
    
    featureset = find_features_NN(post)
    
    result = loaded_model.classify(featureset)
        
    return transformSubreddit(result)

def classifyDepression_NN(post):
    modelFile = './../NoNormalization/Models/finalized_model_is_depression.sav'
    post = cleanPost(post)
    
    loaded_model = pickle.load(open(modelFile, 'rb'))
    
    featureset = find_features_NN(post)
    
    result = loaded_model.classify(featureset)
        
    return transformDepression(result)

# 
# PORTER STEMMER
# 

def cleanStemPost(post):
    urls = re.compile(r'(https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]+\.[^\s]{2,}|www\.[a-zA-Z0-9]+\.[^\s]{2,})')
    numbers = re.compile(r'\d+(\.\d+)?')
    posessivePronouns = re.compile(r"’s")
    apostrophe=re.compile(r"’")
    someSigns =re.compile(r"\\n|\\r")
    punctuation = re.compile(r"[^\w\s]")
    whitespaces = re.compile(r'\s+')
    leadTrailWhitespace = re.compile(r'^\s+|\s+?$')
    
    cleanPost = post.lower()
    cleanPost = urls.sub('url',cleanPost)
    cleanPost = numbers.sub('nmbr',cleanPost)
    cleanPost = posessivePronouns.sub('',cleanPost)
    cleanPost = apostrophe.sub('',cleanPost)
    cleanPost = someSigns.sub('',cleanPost)
    cleanPost = punctuation.sub(' ',cleanPost)
    cleanPost = whitespaces.sub(' ',cleanPost)
    cleanPost = leadTrailWhitespace.sub('',cleanPost)
    
    ps = nltk.PorterStemmer()
    
    # Create an empty list containing Stemmed words
    stemmed_list = []
    
    post_words = cleanPost.split(" ")
    
    # Iterate through every word to Stem
    for word in post_words:
        stemmed_list.append(ps.stem(word))
        
    # Join the list
    stemmed_post = " ".join(stemmed_list)
    
    cleanPost = stemmed_post
    
    # stop words
    with open('./../stop_words_no_punct.data', 'rb') as filehandle:
        # read the data as binary data stream
        stop_words_no_punct = pickle.load(filehandle)
        
    post_without_stop_words = []

    text_tokens = word_tokenize(cleanPost)
    tokens_without_stop_words = [word for word in text_tokens if not word in stop_words_no_punct]
    post_without_stop_words = (" ").join(tokens_without_stop_words)

    cleanPost = post_without_stop_words
    
    return cleanPost

def find_features_PS(post):
    words = word_tokenize(post)
    features = {}
    
    #read generated features for PS
    with open('./../PorterStemmer/word_features.data', 'rb') as filehandle:
        # read the data as binary data stream
        word_features = pickle.load(filehandle)
    
    for word in word_features:
        features[word] = (word in words)
    return features

def transformSubreddit(x):
    switcher={
            0:'r/depression',
            1:'r/unpopularopinion',
            2:'r/lonely',
            3:'r/MachineLearning',
         }
    return switcher.get(x)

def transformDepression(x):
    switcher={
            0:'from r/depression',
            1:'not from r/depression'
         }
    return switcher.get(x)

def classifySubreddit_PS(post):
    modelFile = './../PorterStemmer/Models/finalized_model_subreddits.sav'
    post = cleanStemPost(post)
    
    loaded_model = pickle.load(open(modelFile, 'rb'))
    
    featureset = find_features_PS(post)
    
    result = loaded_model.classify(featureset)
        
    return transformSubreddit(result)

def classifyDepression_PS(post):
    modelFile = './../PorterStemmer/Models/finalized_model_is_depression.sav'
    post = cleanStemPost(post)
    
    loaded_model = pickle.load(open(modelFile, 'rb'))
    
    featureset = find_features_PS(post)
    
    result = loaded_model.classify(featureset)
        
    return transformDepression(result)

# 
# WORD NET LEMMATIZER
# 

def cleanLemmatizePost(post):
    urls = re.compile(r'(https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]+\.[^\s]{2,}|www\.[a-zA-Z0-9]+\.[^\s]{2,})')
    numbers = re.compile(r'\d+(\.\d+)?')
    posessivePronouns = re.compile(r"’s")
    apostrophe=re.compile(r"’")
    someSigns =re.compile(r"\\n|\\r")
    punctuation = re.compile(r"[^\w\s]")
    whitespaces = re.compile(r'\s+')
    leadTrailWhitespace = re.compile(r'^\s+|\s+?$')
    
    cleanPost = post.lower()
    cleanPost = urls.sub('url',cleanPost)
    cleanPost = numbers.sub('nmbr',cleanPost)
    cleanPost = posessivePronouns.sub('',cleanPost)
    cleanPost = apostrophe.sub('',cleanPost)
    cleanPost = someSigns.sub('',cleanPost)
    cleanPost = punctuation.sub(' ',cleanPost)
    cleanPost = whitespaces.sub(' ',cleanPost)
    cleanPost = leadTrailWhitespace.sub('',cleanPost)
    
    wordnet_lemmatizer = WordNetLemmatizer()
    
    # Create an empty list containing Stemmed words
    lemmatized_list = []
    
    post_words = cleanPost.split(" ")
    
    # Iterate through every word to Stem
    for word in post_words:
        lemmatized_list.append(wordnet_lemmatizer.lemmatize(word, pos="v"))
        
    # Join the list
    lemmatized_post = " ".join(lemmatized_list)
    
    cleanPost = lemmatized_post
    
    # stop words
    with open('./../stop_words_no_punct.data', 'rb') as filehandle:
        # read the data as binary data stream
        stop_words_no_punct = pickle.load(filehandle)
        
    post_without_stop_words = []

    text_tokens = word_tokenize(cleanPost)
    tokens_without_stop_words = [word for word in text_tokens if not word in stop_words_no_punct]
    post_without_stop_words = (" ").join(tokens_without_stop_words)

    cleanPost = post_without_stop_words
    
    return cleanPost

def find_features_LM(post):
    words = word_tokenize(post)
    features = {}
    
    #read generated features for Lemmatizer
    with open('./../WordNetLemmatizer/word_features.data', 'rb') as filehandle:
        # read the data as binary data stream
        word_features = pickle.load(filehandle)
    
    for word in word_features:
        features[word] = (word in words)
    return features

def classifySubreddit_LM(post):
    modelFile = './../WordNetLemmatizer/Models/finalized_model_subreddits.sav'
    post = cleanLemmatizePost(post)
    
    loaded_model = pickle.load(open(modelFile, 'rb'))
    
    featureset = find_features_LM(post)
    
    result = loaded_model.classify(featureset)
        
    return transformSubreddit(result)

def classifyDepression_LM(post):
    modelFile = './../WordNetLemmatizer/Models/finalized_model_is_depression.sav'
    post = cleanLemmatizePost(post)
    
    loaded_model = pickle.load(open(modelFile, 'rb'))
    
    featureset = find_features_LM(post)
    
    result = loaded_model.classify(featureset)
        
    return transformDepression(result)

# 
# 
# 

def show_predict_page():
    st.title("Reddit Post Classifier")

    post = st.text_area("Enter Post")

    classify = st.button("Classify")
    if classify:
        classifyResult_NN = classifySubreddit_NN(post)
        depressionResult_NN = classifyDepression_NN(post)

        st.header('Using No Normalization:')
        st.subheader('Subreddit classifier:')
        st.write('Post is likely from '+ classifyResult_NN)
        st.subheader('Is r/depression post:')
        st.write('Post is likely '+ depressionResult_NN)

        classifyResult_PS = classifySubreddit_PS(post)
        depressionResult_PS = classifyDepression_PS(post)

        st.header('Using Porter Stemmer:')
        st.subheader('Subreddit classifier:')
        st.write('Post is likely from '+ classifyResult_PS)
        st.subheader('Is r/depression post:')
        st.write('Post is likely '+ depressionResult_PS)

        classifyResult_LM = classifySubreddit_LM(post)
        depressionResult_LM = classifyDepression_LM(post)

        st.header('Using WordNetLemmatizer:')
        st.subheader('Subreddit classifier:')
        st.write('Post is likely from '+ classifyResult_LM)
        st.subheader('Is r/depression post:')
        st.write('Post is likely '+ depressionResult_LM)