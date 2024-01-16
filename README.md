# MOODSIC
---
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

---


