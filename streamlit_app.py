import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

class SpotifyRecommender:
    def __init__(self, rec_data):
        self.rec_data_ = rec_data

    def change_data(self, rec_data):
        self.rec_data_ = rec_data

    def get_recommendations(self, song_name, amount=1):
        distances = []
        song = self.rec_data_[(self.rec_data_['Song Name'].str.lower() == song_name.lower())].head(1)
        if song.empty:
            return pd.DataFrame()  # Return an empty DataFrame if the song is not found
        song = song.iloc[0]
        res_data = self.rec_data_[self.rec_data_['Song Name'].str.lower() != song_name.lower()]
        
        numeric_cols = self.rec_data_.select_dtypes(include=[np.number]).columns
        
        for _, r_song in tqdm(res_data.iterrows(), total=res_data.shape[0]):
            dist = 0
            for col in numeric_cols:
                dist += np.absolute(float(song[col]) - float(r_song[col]))
                
            distances.append(dist)
        
        res_data['distance'] = distances
        res_data = res_data.sort_values('distance')
        columns = ['Artist', 'Song Name']
        return res_data[columns][:amount]

# Function to normalize column
def normalize_column(data, col):
    max_d = data[col].max()
    min_d = data[col].min()
    data[col] = (data[col] - min_d) / (max_d - min_d)

st.title('Spotify Song Recommendation System')

# Step 1: Upload the Dataset
st.write("## Upload Dataset")
uploaded_file = st.file_uploader("Choose a CSV file")
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("### Dataset Preview")
    st.write(data.head())

    # Preprocess the data
    data = data.replace(',', ' ')
    num_types = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    num = data.select_dtypes(include=num_types)
    for col in num.columns:
        normalize_column(data, col)

    km = KMeans(n_clusters=10)
    cat = km.fit_predict(num)
    data['cat'] = cat
    normalize_column(data, 'cat')

    recommender = SpotifyRecommender(data)

    # Step 2: Input for Song Title
    st.write("## Song Recommendation")
    song_title = st.text_input("Enter a song title")
    num_recommendations = st.slider("Number of recommendations", min_value=1, max_value=20, value=5)

    # Step 3: Recommendation Button
    if st.button("Get Recommendations"):
        if song_title:
            recommendations = recommender.get_recommendations(song_title, num_recommendations)
            if not recommendations.empty:
                st.write("### Recommendations")
                st.write(recommendations)
            else:
                st.write("Song not found in the dataset.")
        else:
            st.write("Please enter a song title.")
