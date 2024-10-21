# recommender/utils/recommendation_engine.py

import pandas as pd
import numpy as np
import nltk
import re
import warnings
import ssl
from sklearn.preprocessing import MinMaxScaler
from scipy import sparse
import ast
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import process, fuzz

from ..models import UserMovieLike


class MovieRecommender:
    def __init__(self):
        warnings.filterwarnings('ignore')
        # Bypass SSL verification (if necessary)
        try:
            _create_unverified_https_context = ssl._create_unverified_context
        except AttributeError:
            pass
        else:
            ssl._create_default_https_context = _create_unverified_https_context

        # Download NLTK data files
        nltk.download('stopwords')
        nltk.download('wordnet')

        # Load the dataset
        self.df = pd.read_csv('recommender/utils/movie_dataset.csv')  # Ensure this file is in your working directory

        # Select features to use
        self.text_features = [
            'keywords', 'cast', 'genres', 'director', 'crew',
            'production_companies', 'production_countries'
        ]
        self.numeric_features = ['vote_count', 'vote_average', 'popularity']

        # Handle missing values
        # Remove rows with missing values in selected features
        self.df = self.df.dropna(subset=self.text_features + self.numeric_features)
        self.df['vote_average'] = pd.to_numeric(self.df['vote_average'], errors='coerce')
        self.df['popularity'] = pd.to_numeric(self.df['popularity'], errors='coerce')
        # Apply text cleaning to each text feature
        for feature in self.text_features:
            self.df[feature] = self.df[feature].apply(self.clean_text)

        # List of available genres (lowercase for matching)
        self.available_genres = [
            'action', 'adventure', 'animation', 'comedy', 'crime', 'documentary',
            'drama', 'family', 'fantasy', 'history', 'horror', 'music',
            'mystery', 'romance', 'science fiction', 'tv movie', 'thriller',
            'war', 'western'
        ]

        # Standardize genres
        self.df['genres'] = self.df['genres'].apply(self.standardize_genres)
        # Convert 'release_date' to datetime
        self.df['release_date'] = pd.to_datetime(self.df['release_date'], errors='coerce')

        # Combine text features into a single string
        self.df['combined_text'] = self.df.apply(self.combine_text_features, axis=1)

        # Create TF-IDF vectorizer for text features
        self.tfidf_vectorizer = TfidfVectorizer()
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.df['combined_text'])

        # Normalize and scale numeric features
        self.df[self.numeric_features] = self.df[self.numeric_features].astype(float)
        scaler = MinMaxScaler()
        self.df[self.numeric_features] = scaler.fit_transform(self.df[self.numeric_features])

        # Create a sparse matrix for numeric features
        self.numeric_matrix = sparse.csr_matrix(self.df[self.numeric_features].values)

        # Combine text and numeric feature matrices
        self.feature_matrix = sparse.hstack([self.tfidf_matrix, self.numeric_matrix])

        # Compute cosine similarity matrix
        self.cosine_sim = cosine_similarity(self.feature_matrix)

        # Normalize movie titles by removing articles and lowercasing
        self.df['normalized_title'] = self.df['title'].apply(self.normalize_title)

    # Function to clean text data
    def clean_text(self, text):
        if isinstance(text, str):
            # Convert text to lowercase
            text = text.lower()
            # Remove non-alphabetic characters
            text = re.sub(r'[^a-zA-Z\s]', '', text)
            # Tokenize text
            words = text.split()
            # Remove stopwords and perform lemmatization
            stop_words = set(stopwords.words('english'))
            lemmatizer = WordNetLemmatizer()
            words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
            # Join the words back into one string
            return ' '.join(words)
        else:
            return ''

    # Standardize genres
    def standardize_genres(self, text):
        if isinstance(text, str):
            text = text.lower()
            words = text.split()
            genres_found = set()

            # Maximum number of words in a genre (e.g., "science fiction" has 2 words)
            max_genre_length = max(len(genre.split()) for genre in self.available_genres)

            # Use a sliding window to find matching genres
            for window_size in range(1, max_genre_length + 1):
                for i in range(len(words) - window_size + 1):
                    phrase = ' '.join(words[i:i + window_size])
                    if phrase in self.available_genres:
                        genres_found.add(phrase.title())

            # Return the genres found as a comma-separated string
            return ','.join(genres_found)
        else:
            return ''

    # Combine text features into a single string
    def combine_text_features(self, row):
        return ' '.join([row[feature] for feature in self.text_features])

    # Normalize movie titles by removing articles and lowercasing
    def normalize_title(self, title):
        # Remove articles
        articles = {'the', 'a', 'an'}
        words = title.lower().split()
        normalized_words = [word for word in words if word not in articles]
        normalized_title = ' '.join(normalized_words)
        return normalized_title

    # Helper function to get movie title from index
    def get_title_from_index(self, index):
        return self.df.iloc[index]['title']

    # Helper function to get index from movie title
    def get_index_from_title(self, user_input_title, threshold=70):
        # Normalize user input
        normalized_input = self.normalize_title(user_input_title)
        # Get the list of normalized titles
        titles = self.df['normalized_title'].tolist()
        # Use RapidFuzz to find the best match along with the similarity score
        match = process.extractOne(normalized_input, titles, scorer=fuzz.token_set_ratio)
        if match:
            best_match_title = match[0]
            similarity_score = match[1]
            if similarity_score >= threshold:
                # Get the index of the best match
                index = self.df[self.df['normalized_title'] == best_match_title].index[0]
                original_title = self.df.loc[index, 'title']
                return index, original_title  # Return index and the original title
            else:
                # Similarity score is below the threshold; consider the movie not found
                return None, None
        else:
            # No matches found
            return None, None
    # Recommendation method with optional filters and sorting
    # recommender/utils/recommendation_engine.py

    def recommend(self, movie_user_likes, num_recommendations=5, genres=None, sort_by='relevance', threshold=70):
        movie_index, matched_title = self.get_index_from_title(movie_user_likes, threshold=threshold)

        if movie_index is not None:
            # Get similarity scores for the input movie
            similarity_scores = list(enumerate(self.cosine_sim[movie_index]))

            # Exclude the input movie itself
            similarity_scores = similarity_scores[1:]

            # Apply filters
            filtered_movies = self.apply_filters(similarity_scores, genres)

            # Apply sorting to the filtered similar movies
            sorted_movies = self.sort_movies(filtered_movies, sort_by)

            # Get top N recommendations
            recommendations = []
            i = 0
            for element in sorted_movies:
                recommended_movie = self.df.iloc[element[0]]
                recommendations.append({
                    'title': recommended_movie['title'],
                    'homepage': recommended_movie.get('homepage', ''),
                    'genres': recommended_movie.get('genres', ''),
                    'release_date': recommended_movie.get('release_date', ''),
                    'overview': recommended_movie.get('overview', ''),
                    'rating': recommended_movie.get('vote_average', ''),
                    'popularity': recommended_movie.get('popularity', ''),
                })
                i += 1
                if i >= num_recommendations:
                    break
            return recommendations, matched_title
        else:
            # Movie not found
            return [], None

    # recommender/utils/recommendation_engine.py

    def recommend_for_user(self, user, num_recommendations=5, genres=None, sort_by='relevance'):
        # Get the list of movies the user has liked
        liked_ids = UserMovieLike.objects.filter(user=user).values_list('movie__id', flat=True)
        if not liked_ids:
            # If the user hasn't liked any movies, fall back to default recommendations
            return self.recommend('')

        # Get indices of liked movies in self.df
        liked_indices = self.df[self.df['id'].isin(liked_ids)].index.tolist()

        # If none of the liked movies are in the dataset, fall back to default recommendations
        if not liked_indices:
            return self.recommend('')

        # Compute the average similarity scores across liked movies
        sim_scores = np.mean(self.cosine_sim[liked_indices], axis=0)
        # Get movie indices and similarity scores
        similar_movies = list(enumerate(sim_scores))

        # Exclude movies the user has already liked
        similar_movies = [(idx, score) for idx, score in similar_movies if idx not in liked_indices]

        # Apply filters
        filtered_movies = self.apply_filters(similar_movies, genres)

        # Apply sorting
        sorted_movies = self.sort_movies(filtered_movies, sort_by)

        # Get top N recommendations
        recommendations = []
        i = 0
        for element in sorted_movies:
            recommended_movie = self.df.iloc[element[0]]
            recommendations.append({
                'id': recommended_movie['id'],
                'title': recommended_movie['title'],
                'homepage': recommended_movie.get('homepage', ''),
                'genres': recommended_movie.get('genres', ''),
                'release_date': recommended_movie.get('release_date', ''),
                'overview': recommended_movie.get('overview', ''),
                'rating': recommended_movie.get('vote_average', ''),
                'popularity': recommended_movie.get('popularity', ''),
            })
            i += 1
            if i >= num_recommendations:
                break
        return recommendations

    # Method to apply filters
    def apply_filters(self, movies, genres):
        filtered_movies = []
        for idx, sim_score in movies:
            movie = self.df.iloc[idx]
            # Filter by genres
            if genres:
                # Get the movie genres as a set
                movie_genres = set([genre.strip().title() for genre in movie['genres'].split(',')])
                selected_genres = set([genre.title() for genre in genres])
                if not movie_genres.intersection(selected_genres):
                    continue
            filtered_movies.append((idx, sim_score))
        return filtered_movies

    # Method to sort movies
    # recommender/utils/recommendation_engine.py

    def sort_movies(self, movies, sort_by):
        if sort_by == 'relevance':
            # Sort by similarity score (descending)
            movies = sorted(movies, key=lambda x: x[1], reverse=True)
        elif sort_by == 'popularity':
            movies = sorted(movies, key=lambda x: self.df.iloc[x[0]]['popularity'], reverse=True)
        elif sort_by == 'rating':
            movies = sorted(movies, key=lambda x: self.df.iloc[x[0]]['vote_average'], reverse=True)
        elif sort_by == 'release_date_newest':
            movies = sorted(movies, key=lambda x: self.df.iloc[x[0]]['release_date'], reverse=True)
        elif sort_by == 'release_date_oldest':
            movies = sorted(movies, key=lambda x: self.df.iloc[x[0]]['release_date'])
        elif sort_by == 'title':
            movies = sorted(movies, key=lambda x: self.df.iloc[x[0]]['title'])
        return movies

    # Optional method to get genres of a movie
    def get_movie_genres(self, movie_title):
        index = self.df[self.df['title'] == movie_title].index[0]
        return self.df.loc[index, 'genres']

    def get_movies_by_ids(self, ids):
        # Get the movies from the dataframe where 'id' is in ids
        movies_df = self.df[self.df['id'].isin(ids)]
        movies = []
        for _, row in movies_df.iterrows():
            movies.append({
                'id': row['id'],
                'title': row['title'],
                'homepage': row.get('homepage', ''),
                'genres': row.get('genres', ''),
                'release_date': row.get('release_date', ''),
                'overview': row.get('overview', ''),
                'rating': row.get('vote_average', ''),
                'popularity': row.get('popularity', ''),
            })
        return movies
