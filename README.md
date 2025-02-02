# Movie_Recomendation_Model
Movie_Recomendation_Model
Movie Recommendation Model

Project Description
This project provides a movie recommendation system based on collaborative filtering using Singular Value Decomposition (SVD). The model predicts which movies a user is likely to enjoy based on historical rating data.

Link to model on Hugging Face: (https://huggingface.co/JuliaBiwan/movie_recomendations)

How to use the model

1. Before running the model, install these dependencies:

pip install numpy pandas scikit-learn 

2. Load the model
You can download the model file (svd_model.pkl) from the Hugging Face repository and load it as follows:

import pickle

Load the model
with open("svd_model.pkl", "rb") as f:
    model = pickle.load(f)

3. Get movie recommendations
To get recommendations for a specific user, use the following code:

def get_movie_recommendations(user_id, rated_movies, top_n=5):
    all_movies = movies_df['Series_Title'].values
    user_ratings = rated_movies[rated_movies[user_id] > 0]
    rated_movies_list = user_ratings['Series_Title'].values
    unrated_movies = [movie for movie in all_movies if movie not in rated_movies_list]

    predictions = []
    for movie in unrated_movies:
        pred = svd_model.predict(user_id, movie)
        predictions.append((movie, pred.est))
    predictions.sort(key=lambda x: x[1], reverse=True)
    top_movies = predictions[:top_n]

    return top_movies

Example usage
user_id = 123  # Replace with a valid user ID
movie_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # Replace with actual movie IDs
print("Recommended movies:", get_recommendations(user_id, model, movie_list))

4. Running in Google Colab

!wget https://huggingface.co/JuliaBiwan/movie_recomendations/resolve/main/svd_model.pkl

import pickle

with open("svd_model.pkl", "rb") as f:
    model = pickle.load(f)

Training data
The model was trained on a dataset containing user-movie ratings. The dataset includes ratings from users on various movies, which are used to generate recommendations based on collaborative filtering.
