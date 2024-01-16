# M00DSIC

**M00DSIC** is a ML web app that classifies and breaks down a given public Spotify collection into Energetic or Non Energetic mood. It displays charts for the collection alongside the BillBoard Year End chart of 2023 for comparison. It shows detailed charts for important audio features of both collections. 

**M00DSIC** uses Support Vector Machines with a Radial kernel.

## Use Cases

 We believe this web app has uses for anyone:
- who is in the Music Industry and needs a top level overview of songs that public likes vs. another collection.
- does Music industry research. If one uses 2023 year as a base year to compare with another collection containing BB Year End songs of previous years or future years.
- looking for Fun. Compare how their collections differ from the top songs of 2023. 
- wants to get a feature breakdown of artists, albums or playlists.

## How to download and use code repo

1. Download the codebase using Git.
2. Download dependencies using the requirements.txt file.
3. It is suggested you to create a virtual environment and download the dependencies in the venv (Virtual Env), as follows:

   `python3 -m venv path_to_file_outside_the_repo`

   `source path_to_file_outside_the_repo/bin/activate`
 4. Install the file in the code repo using `pip3 install -r requirements.txt`.
 5. We used our own API keys to pull metadata from Spotify Web API. PLease use your own API keys by going to [link](https://developer.spotify.com/) and creating your own app.
 6. Store the API keys in a **secrets.toml** file within ./streamlit directory in the repo.
 7. To run the app: `streamlit run app.py`.


## What do some of these files mean?
- `Moods.txt`: Has all the IDs of the playlists we used to create our training dataset.
- `spotifytrackinfo.csv`: Our training dataset
- `model.pk1`: Pickle file containing model, features and scaler.
- `bbyearend.csv`: Dataset of the BillBoard Year End Hot 100 2023 songs.
- `visuals.pk1`: Pickle containing the charts used in the web app.
- `.streamlit/config.toml`: Configuration file for the web app.
- `Theory.md`: Knowledge base for our project - what we learned and references.

