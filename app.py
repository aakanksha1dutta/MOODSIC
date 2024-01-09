#import dependencies
import streamlit as st
import os
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from dotenv import load_dotenv, dotenv_values
import re
import pickle
import pandas as pd
import numpy as np


#api keys
client_id = st.secrets['CLIENT_ID']
client_secret = st.secrets['CLIENT_SECRET']
redirect_url = st.secrets['REDIRECT_URL']

#helper function
def extract_uri(url):
    pattern = r'(track|album|artist|playlist)/([a-zA-Z0-9]{22})' #spotify uri is a 22 char alphanumeric identifier
    
    # Try to find a match in the URL
    match = re.search(pattern, url)

    # If a match is found, return the Spotify ID
    if match:
        return match.group(2)
    else:
        return None

def partition(lst, size):
    split = [lst[i:i+size] for i in range(0,len(lst),size)]
    return split







#web app

st.set_page_config(
    page_title="MOODSIC",
    page_icon="🎧",
    initial_sidebar_state="expanded",
    layout="wide"
)

#sidebar
with st.sidebar:
    st.title("Make a selection! :zap:")
    method = st.selectbox("Choose your collection",('Playlist', 'Album', "Artist's Top 10"))
    link = st.text_input("Enter your Spotify link here:")
    st.caption("Your collection must be publicly accessible!")
    go = st.button("GO")



st.title(":green[M00DSIC]")
st.header("Check your music's energy and more", divider="green")
st.write('**M00DSIC** takes in any Spotify link to a public collection, be it an album, playlist or an artist, \
         and shows you simple visualization that breaks down your collection into its bare components.')

st.write('Powered by a machine learning algorithm model called Support Vector Machines and Spotify Web API, **M00DSIC**\
         delivers audio features about your playlist. Even better, you get to see a side by side comparison\
         of your collection with  *Billboard Year End Hot 100 Chart of 2023*.')
st.markdown('`Note: Due to Spotify API limits, M00DSIC can only read first 100 tracks of a playlist or an album and Top 10 tracks\
            of an artist. Since this is a hobby project, we are limiting our API calls to the allowed limits.`')

#Add visualization 

#in action
if(go):

    #initiate session on GO
    auth_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
    sp = spotipy.Spotify(auth_manager=auth_manager)

    #load model
    file = pickle.load(open("model.pk1","rb"))
    model = file['model']
    feature_list = file['features']
    scaler = file['scaler']

    #helper functions --------------------------------------------------------------------------------
    def preprocess(df):
        from sklearn.preprocessing import MinMaxScaler
        features = df[feature_list]
        scaled_df = scaler.transform(features)
        features = pd.DataFrame(scaled_df, columns=features.columns)
        return features
    
    def get_playlist_tracks(playlist_id):
        track_ids = []
        track_names =[]
        track_artists = []
        tracks = sp.playlist_tracks(playlist_id)['items']
        for track in tracks:
            track_ids.append(track['track']['id'])
            track_names.append(track['track']['name'])
            track_artists.append(track['track']['artists'])
        return track_ids, track_names, track_artists
    
    def get_album_tracks(album_id):
        track_ids = []
        track_names = []
        track_artists = []
        tracks = sp.album_tracks(album_id)['items']
        for track in tracks:
            track_ids.append(track['id'])
            track_names.append(track['name'])
            track_artists.append(track['artists'])
        return track_ids, track_names, track_artists
    
    def get_artist_tracks(artist_id):
        track_ids = []
        track_names = []
        track_artists = []
        tracks = sp.artist_top_tracks(artist_id)['tracks']
        for track in tracks:
            track_ids.append(track['id'])
            track_names.append(track['name'])
            track_artists.append(track['artists'])
        return track_ids, track_names, track_artists
    
    def get_features(track_ids):
        part = partition(track_ids,100) #list of partitioned tracks
        audio_features = []
        for p in part:
            audio_features.extend(sp.audio_features(p))
        return audio_features

    def is_energetic(x):
        return 'Energetic' if x==1 else 'Not Energetic'

    def get_class(url, method):
        id = extract_uri(url)
        if method=='playlist':
            track_ids, track_names, track_artists = get_playlist_tracks(id)
        elif method=='album':
            track_ids, track_names, track_artists = get_album_tracks(id)
        elif method=='artist':
            track_ids, track_names, track_artists = get_artist_tracks(id)
            

        #finding first artists
        first_artist = []
        for i in range(0,len(track_artists)):
            first_artist.append(track_artists[i][0]['name'])

        
        #making dataframe
        df = pd.DataFrame(get_features(track_ids))
        df.insert(loc=0, column='track_id', value=track_ids)
        df.insert(loc=1, column='track_name', value=track_names)
        df.insert(loc=2, column='first_artist', value=first_artist)

        #extract features
        features = preprocess(df)
        y_pred = model.predict(features)

        df['pred'] = pd.Series(y_pred)
        df['class'] = df['pred'].apply(lambda x: is_energetic(x))

        #we will make a classification report
        energetic_tracks = df[['first_artist','track_name']][df.pred==1]
        prop = len(energetic_tracks)/len(df)

        return {'energetic_tracks': energetic_tracks, 'prop': prop, 'dataframe':df}
    
    #end of helper functions --------------------------------------------------------------------------------
    method = method.split()[0].lower()
    st.write(method)
    #report = get_class(link, method)



    with st.sidebar:
        st.write("in action now")




