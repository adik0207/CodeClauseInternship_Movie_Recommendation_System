import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors

# Load movie ratings data
ratings_data = pd.read_csv('C:/Users/hp/Downloads/archive (2)/Dataset.csv')
movies_data = pd.read_csv('C:/Users/hp/Downloads/archive (2)/Movie_Id_Titles.csv')

# Merge ratings and movies data
movie_ratings = pd.merge(ratings_data, movies_data, on='item_id')

user_movie_ratings = movie_ratings.pivot_table(index='user_id', columns='title', values='rating')

user_movie_ratings = user_movie_ratings.fillna(0)

item_similarity = cosine_similarity(user_movie_ratings.T)

movie_similarity_df = pd.DataFrame(item_similarity, index=user_movie_ratings.columns, columns=user_movie_ratings.columns)

# Function to get movie recommendations for a given movie
def get_movie_recommendations(movie_name, num_recommendations=5):
    similar_movies = movie_similarity_df[movie_name]
    recommended_movies = similar_movies.sort_values(ascending=False)[1:num_recommendations+1]
    return recommended_movies

# Function to get user recommendations based on user's watched movies
def get_user_recommendations(user_id, num_recommendations=5):
    user_ratings = user_movie_ratings.loc[user_id]
    user_watched_movies = user_ratings[user_ratings > 0].index
    recommended_movies = pd.Series()

    for movie in user_watched_movies:
        similar_movies = get_movie_recommendations(movie)
        recommended_movies = recommended_movies.append(similar_movies)

    recommended_movies = recommended_movies.groupby(recommended_movies.index).sum()
    recommended_movies = recommended_movies.sort_values(ascending=False)

    # Filter out already watched movies
    recommended_movies = recommended_movies.drop(user_watched_movies, errors='ignore')

    return recommended_movies.head(num_recommendations)

# Example usage:
user_id = 1
recommended_movies = get_user_recommendations(user_id)
print(f"Recommended movies for user {user_id}:")
print(recommended_movies)
