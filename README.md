# SubRedditClassification

## Contains trained models for subreddit classification. 
- Classifies a post to one of four subreddits (**r/depression**, **r/lonely**, **r/unpopularopinion**, **r/MachineLearning**);
- Classifies whether a post belongs to **r/depression**.

### Models were trained:
- Using **no normalization**
- Using the **Porter Stemmer**
- Using **WordNetLemmatizer**

Appropriately named folders contain the models. Each folder contains **CleanDataAndTrainModel** file with all posts processing and model training. **ClassifyFunction** contains function for classifying one post with the models. Training, testing  data and trained models are in **/Model**

####  Data folder
Contains scraped posts from Reddit from the four subreddits and a file with all posts combined.

#### The app 
Main directory contains the Streamlit app with post classifier page, the four subreddit exploration page (using WordNetLemmatizer) and exploration page for the three models.

#### To run the App
> **streamlit run streamlit_app.py**

App also deployed to [https://share.streamlit.io/alliix/subredditclassification/main](https://share.streamlit.io/alliix/subredditclassification/main)
