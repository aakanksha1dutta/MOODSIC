# Theory, Thoughts and Learnings

We are creating this Theory.md file to serve as a central knowledge base behind what we did in this project so anyone wishing to start out with something similar can learn from it. 

## Music Mood Theory

- References:
    * [Mood Classification Handbook](https://sites.tufts.edu/eeseniordesignhandbook/2015/music-mood-classification/): This handbook taught us about Robert Thayer’s traditional model of mood. To quote:

        > "The model divides songs along the lines of energy and stress, from happy to sad and calm to energetic, respectively (Bhat et al 359). The eight categories created by Thayer’s model include the extremes of the two lines as well as each of the possible intersections of the lines (e.g. happy-energetic or sad-calm)."
        
    * [Machine Learning for Mood Classification by neokt](https://neokt.github.io/projects/audio-music-mood-classification/): Ting Neo attempted Music Mood Classification with a triangular method of classification where she predicted arousal and positivity scale of a song and then use those to zone in on mood classes. We refered to her detailed document for understanding of audio features and the mood model.


## Learnings along the way

1. **Multi Class Classification** : We had initially attempted a multi-class music mood classification with 4 classes (Happy, Sad, Energetic, Calm) where we tried classifiers like Logistic Regression, Support Vector Machines, Random Forests and kNN. But since music mood may overlap in the traditional model, our models weren't just good enough predicting the class. Music does not fall in 4 separate moods but can have multiple moods at the same time. Thus,
    * We realized we can only do a binary classification with Happy/Sad or Energetic/Non Energetic. 
    * For more advanced mood prediction, a *Multi-Label* classification and not *Multi-Class* classification could be implemented. Here, mood are labels and tracks can have multiple labels. Think of them like tags that describe the song mood.

    We ended up doing *Binary Classification*.

2. **Different Scalers** : We used different scalers to scale our features at first. This was a grave error since that meant our features for our training sample and the use cases sample were scaled very differently. 
    * So, we fit a scaler to our training sample and then pickled it with our model in `model.pk1`, to use with other datasets..

3. **Unbalanced dataset**: Since 1/4th of our training dataset was Energetic, we could either drop some rows (undersampling) of Non Energetic or get more samples of Energetic(oversampling). We chose to do the latter since our dataset was already small. 
    * We used SMOTE (Synthetic Minority Oversampling Technique) from the `imblearn` library. 
    * Since there is a possibility of overfitting using SMOTE, we used f1-score, precision, recall for both classes as a marker of our model performance instead of accuracy (see, [Accuracy Paradox](https://en.wikipedia.org/wiki/Accuracy_paradox#:~:text=The%20accuracy%20paradox%20is%20the,too%20crude%20to%20be%20useful.))


## Thoughts
1. We created a web app which takes in publicly accessible collections. To take user collections, we would need to trigger a login page where user can login to authorize Spotify to give us their data, using OAuth2.0 framework. 
    - Streamlit does not have this feature. One can use Django or Flask for more full fledged Web Apps.
    
