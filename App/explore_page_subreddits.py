import streamlit as st
import pickle
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import chi2
import numpy as np
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud
import matplotlib.pyplot as plt
nltk.download('webtext')
import matplotlib.pyplot as plt

def getWordFrequency(processedPostOutputFile):
    posts = pd.read_csv(processedPostOutputFile)

    allPostsConcat = ''
    for post in posts.Post_Parsed:
        if(type(post)==str):
            allPostsConcat+=post

    # create bag-of-words
    all_words = []

    words = word_tokenize(allPostsConcat)
    for word in words:
        all_words.append(word)

    all_words = nltk.FreqDist(all_words)
    
    return all_words



def show_explore_page_subreddits():
    st.title("Explore Subreddits")
    subreddits = [{'title': 'r/depression', 'file': './WordNetLemmatizer/depression_posts_processed.csv'},
    {'title': 'r/lonely', 'file': './WordNetLemmatizer/lonely_posts_processed.csv'},
    {'title': 'r/unpopularopinion', 'file': './WordNetLemmatizer/unpopularopinion_posts_processed.csv'},
    {'title': 'r/MachineLearning', 'file': './WordNetLemmatizer/machinelearning_posts_processed.csv'}]

    for subreddit in subreddits:
        st.header(subreddit['title'])
        freqWords = getWordFrequency(subreddit['file'])
        st.write('Number of words: {}'.format(len(freqWords)))
        st.write('Most common words: {}'.format(freqWords.most_common(50)))

        data_analysis = freqWords
        filter_words = dict([(m, n) for m, n in data_analysis.items() if len(m) > 2])
        wcloud = WordCloud().generate_from_frequencies(filter_words)

        # Plotting the wordcloud
        fig, ax = plt.subplots()
        ax.imshow(wcloud, interpolation="bilinear")
        ax.axis("off")
        (-0.5, 399.5, 199.5, -0.5)
        st.pyplot(fig)