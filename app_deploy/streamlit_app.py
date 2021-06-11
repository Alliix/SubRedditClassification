import streamlit as st
from predict_page import show_predict_page
from explore_page_NN import show_explore_page_NN
from explore_page_LM import show_explore_page_LM
from explore_page_PS import show_explore_page_PS
from explore_page_subreddits import show_explore_page_subreddits

page = st.sidebar.selectbox("Choose page", ("Predict", "Explore model Without Normalization", "Explore Porter Stemmer model", "Explore WordNetLemmatizer model", "Show Subreddit Data (Lemmatizer)"))

if page == "Predict":
    show_predict_page()
if page == "Explore model Without Normalization":
    show_explore_page_NN()
if page == "Explore Porter Stemmer model":
    show_explore_page_PS()
if page == "Explore WordNetLemmatizer model":
    show_explore_page_LM()
if page == "Show Subreddit Data (Lemmatizer)":
    show_explore_page_subreddits()