import streamlit as st
import nltk
from nltk.tokenize import word_tokenize
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt

nltk.download('webtext')

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

@st.cache(allow_output_mutation=True)
def getDepressionFreqWords():
    return getWordFrequency('depressed_posts_processed.csv')

def show_explore_page():
    st.title("Explore Subreddit Posts")

    depressionFreqWords = getDepressionFreqWords()
    st.write('Number of words in r/depression posts: {}'.format(len(depressionFreqWords)))
    
    data_analysis = depressionFreqWords
 
    filter_words = dict([(m, n) for m, n in data_analysis.items() if len(m) > 2])
    
    wcloud = WordCloud().generate_from_frequencies(filter_words)
    
    # Plotting the wordcloud
    fig, ax = plt.subplots()
    ax.imshow(wcloud, interpolation="bilinear")
    
    ax.axis("off")
    (-0.5, 399.5, 199.5, -0.5)
    st.pyplot(fig)