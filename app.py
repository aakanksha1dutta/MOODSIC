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
import seaborn as sns
import matplotlib.pyplot as plt


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
    method = st.selectbox("Choose your collection",('Playlist', 'Album', "Artist Top 10"))
    link = st.text_input("Enter your Spotify link here:")
    st.caption("Your collection must be publicly accessible!")
    go = st.button("GO")



st.title(":green[M00DSIC]")
st.header("Check your music's energy and more", divider="green")
st.write('**M00DSIC** takes in any Spotify link to a public collection, be it an album, playlist or an artist, \
         and shows you simple visualization that breaks down your collection into its bare components.')

st.write('Powered by a machine learning model called Support Vector Machines and Spotify Web API, **M00DSIC**\
         delivers audio features about your playlist. Even better, you get to see a side by side comparison\
         of your collection with  *Billboard Year End Hot 100 Chart of 2023*.')
st.markdown('`Note: Due to Spotify API limits, M00DSIC can only read first 100 tracks of a playlist or an album and Top 10 tracks\
            of an artist. Since this is a hobby project, we are limiting our API calls to the allowed limits.`')
st.divider()
main_area = st.empty()


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
        try:
            if method=='playlist':
                track_ids, track_names, track_artists = get_playlist_tracks(id)
            elif method=='album':
                track_ids, track_names, track_artists = get_album_tracks(id)
            elif method=='artist':
                track_ids, track_names, track_artists = get_artist_tracks(id)
        except:
            st.sidebar.write("Choose a valid method and a publicly accessible playlst/album/artist URL!")
            return
            

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
    
    def piechart(df1, df2): 
        #df1 is energetic, df2 is whole df   
        # calculating percentage of songs that were energetic
        prop = len(df1)/len(df2)*100
        print(prop)
        data = [prop,(100-prop)]
        labels = ['Energetic','Not Energetic']
        colors = ["#2E8B57", "#646467"]
        fig, axs = plt.subplots()
        fig.set_facecolor('#0F0F0F')
        axs.pie(data, labels=labels, labeldistance=0.4, autopct='%.0f%%', wedgeprops = {"edgecolor" : "black", 'linewidth': 2, 'antialiased': True}, colors = colors, 
        textprops=dict(color='w',weight='bold')) 
        axs.legend(labels)
        return fig
    
    def plot_tempo(df):
        fig, axs = plt.subplots()
        sns.kdeplot(data = df, x='tempo', color="#2E8B57", ax=axs)
        plt.xlim(0,300)
        fig.set_facecolor('#0F0F0F')
        axs.set_facecolor('#0F0F0F')  
        axs.spines['bottom'].set_color('white')
        axs.spines['left'].set_color('white')
        axs.tick_params(color='white', labelcolor='white')
        axs.set_xlabel('Tempo / BPM').set_color('white')
        axs.set_ylabel('Probability Density Function').set_color('white')
        return fig
    
    def plot_loudness(df):
        fig, axs = plt.subplots()
        sns.kdeplot(data = df, x = 'loudness', color="#2E8B57")
        fig.set_facecolor('#0F0F0F')
        axs.set_facecolor('#0F0F0F')  
        axs.spines['bottom'].set_color('white')
        axs.spines['left'].set_color('white')
        axs.tick_params(color='white', labelcolor='white')
        axs.set_xlabel('Loudness / dB').set_color('white')
        axs.set_xlim(-60,0)
        axs.set_ylabel('Probability Density Function').set_color('white')
        return fig
    
    def plot_mode(df):
        colours = ["seagreen", "#646467"]
        fig, axs = plt.subplots()
        sns.countplot(df, x="mode", stat="percent", hue="mode", palette=colours, ax=axs)
        plt.legend(labels = ['minor', 'major'])
        fig.set_facecolor('#0F0F0F')
        axs.set_facecolor('#0F0F0F')  
        axs.spines['bottom'].set_color('white')
        axs.spines['left'].set_color('white')
        axs.tick_params(color='white', labelcolor='white')
        axs.set_xlabel('Scale').set_color('white')
        axs.set_ylabel('Percentage').set_color('white')
        return fig

    def plot_key (df):
        fig, axs = plt.subplots()
        colors = sns.light_palette("seagreen", as_cmap=True)
        sns.countplot(df, x = "key", stat="percent", hue="key", legend=False, palette=colors, ax=axs)
        fig.set_facecolor('#0F0F0F')
        axs.set_facecolor('#0F0F0F')  
        axs.spines['bottom'].set_color('white')
        axs.spines['left'].set_color('white')
        axs.tick_params(color='white', labelcolor='white')
        axs.set_xlabel('Key (Pitch Class Notation)').set_color('white')
        axs.set_ylabel('Percentage').set_color('white')    
        return fig
    
    def plot_otherFeatures(df):
        features_adj = df.drop(['track_id', 'track_name', 'first_artist', 'key', 'mode', 'loudness','tempo', 'uri', 'type',
                                            'id','track_href', 'analysis_url', 'duration_ms', 'time_signature','pred'], axis=1)
        fig, axs = plt.subplots()
        colors = sns.light_palette("seagreen",7)
        fig.set_facecolor('#0F0F0F')
        axs.set_facecolor('#0F0F0F')  
        axs.spines['bottom'].set_color('white')
        axs.spines['left'].set_color('white')
        axs.tick_params(color='white', labelcolor='white')
        axs.set_xlabel('Value').set_color('white')
        axs.set_xlim(0,1)
        axs.set_ylabel('Track Features').set_color('white')
        sns.barplot(data = features_adj, orient="y", errorbar=None, palette=colors, ax=axs)
        return fig
    
    #end of helper functions --------------------------------------------------------------------------------
    method = method.split()[0].lower()
    report = get_class(link, method)

    if report:
        main_area.text("\n \n")

        #set the variables
        user_energetic_tracks = report['energetic_tracks']
        user_prop = report['prop']*100
        user_df = report['dataframe']

   
        bb_data = pickle.load(open("visuals.pk1", "rb"))
        bb_prop = bb_data['prop']
        

        ratio = ((user_prop-bb_prop)/bb_prop)*100

        #display
        with main_area.container():

            with st.expander("What just happened? :flushed:"):
                st.write("**MOODSIC** took at most 100 songs of your given collection (or top 10 songs of an artist) and produced charts that breaks the audio profile of your songs.")
                st.write("Expand the notes under each chart for a brief explanation on each feature.")
                st.write("Compare your charts (left) with the BillBoard Hot 100 Year End songs from 2023 (right).")
            st.divider()

            st.metric(label = ":green[%] of Energetic Tracks in Collection",value = "{:.2f}%".format(user_prop), delta="{:.2f}%".format(ratio))
            st.caption("* ↑↓ shows the percentage difference between your collection and BB chart")

            col1, col2 = st.columns(2)

            with col1:
                st.header(":green[Your Collection] Breakdown")

                st.subheader(":green[Energetic] Tracks")
                st.dataframe(user_energetic_tracks, hide_index=True, column_config={'first_artist':'First Artist','track_name':'Track'}, height = 500)
                st.expander("What does this mean?").write(
                    "This table shows a list of all energetic songs in your playlist. Click on full screen to enlarge it or download button to download the \
                        full table as a .csv file."
                )

                st.subheader(":green[Composition]")
                st.pyplot(piechart(user_energetic_tracks,user_df))
                st.expander("What does this mean?").markdown(
                    "Shows the **percentage** of songs in your collection that are classified as energetic or not."
                )

                st.subheader(":green[Loud]ness")
                st.pyplot(plot_loudness(user_df))
                st.expander("What does this mean?").markdown(
                    "Loudness strongly influences the liveliness of a track, \
                        so louder songs are perceived as happier and more energetic songs. \
                        Loudness correlates to physical strength (amplitude). Values typically range between **-60** and **0 db**."
                )
                

                st.subheader(":green[Key]/Pitch")
                st.pyplot(plot_key(user_df))
                with st.expander("What does this mean?"):
                    st.markdown(
                    "The key/pitch the track is in. Integers map to pitches using standard Pitch Class notation, described as below")
                    keys = pd.DataFrame({'Integer': np.arange(0,12),'Pitch':['C#','C♯, D♭', 'D','D♯, E♭','E','F','F♯, G♭','G','G♯, A♭','A','A♯, B♭','B']})
                    keys.set_index('Integer', inplace=True)
                    st.table(keys)
                    st.write("-1 if if no key detected")

                st.subheader(":green[Tempo] (Beats Per Minute)")
                st.pyplot(plot_tempo(user_df))
                st.expander("What does this mean?").write(
                    "Tempo considers the speed or beats per minute of the tracks in the playlist. \
                        High tempos are associated with making songs sound more lively and energetic due to the fast pacing."
                )

                st.subheader("Major vs Minor :green[Scale]")
                st.pyplot(plot_mode(user_df))
                st.expander("What does this mean?").write(
                    "In Western music, generally, \
                        major scales are associated with evoking happier (energetic) emotions and minor scales with sadder emotions.\
                            This countplot shows the scale distribution of the collection."
                )

                st.subheader("Other :green[Features]")
                st.pyplot(plot_otherFeatures(user_df))
                with st.expander("What does this mean?"):
                    st.write(
                    'Spotify measures all the following values on a scale of 0.0 to 1.0.')
                    st.markdown("- *Danceability* describes how suitable a track is for dancing based on a \
                        combination of musical elements including tempo, rhythm stability, beat strength, and overall regularity.")
                    st.markdown("- *Energy* represents a perceptual measure of intensity and activity. Typically, energetic tracks feel fast, loud, and noisy. \
                            Perceptual features contributing to this attribute include dynamic range, perceived loudness, timbre, onset rate, and general entropy.")
                    st.markdown("- *Speechiness* detects the presence of spoken words in a track. The more exclusively speech-like the recording, the closer to 1.0 the attribute value.")
                    st.markdown("- *Acousticness* is a confidence measure of whether the track is acoustic with 1.0 representing high confidence the track is acoustic.")
                    st.markdown('- *Instrumentalness* predicts whether a track contains no vocals. "Ooh" and "aah" sounds are treated as instrumental in this context.\
                        The closer the instrumentalness value is to 1.0, the greater likelihood the track contains no vocal content.')
                    st.markdown('- *Liveness* detects the presence of an audience in the recording. Higher liveness values represent an increased probability that the track was performed live.')
                    st.markdown('- *Valence* describes the musical positiveness conveyed by a track. Tracks with high valence sound more positive (e.g. happy, cheerful, euphoric), \
                        while tracks with low valence sound more negative (e.g. sad, depressed, angry).')
            
            with col2:
                st.header(":green[BB Year End] Breakdown")
                st.subheader(":green[Energetic] Tracks")
                st.dataframe(bb_data['energetic_tracks'], hide_index=True, column_config={'first_artist':'First Artist','track_name':'Track'}, height = 500)
                st.subheader(":green[Composition]")
                st.pyplot(bb_data['piechart'])
                st.subheader(":green[Loud]ness")
                st.pyplot(bb_data['loudness'])
                st.subheader(":green[Key]/Pitch")
                st.pyplot(bb_data['key'])
                st.subheader(":green[Tempo] (Beats Per Minute)")
                st.pyplot(bb_data['tempo'])
                st.subheader("Major vs Minor :green[Scale]")
                st.pyplot(bb_data['mode'])
                st.subheader("Other :green[Features]")
                st.pyplot(bb_data['otherFeatures'])

    st.divider()    

st.write("Created with :love_letter: by [Aakanksha Dutta](https://github.com/aakanksha1dutta) and [Aabha Pandit](https://github.com/aabpandit)")              





