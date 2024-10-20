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
    def get_index_from_title(self, user_input_title):
        # Normalize user input
        normalized_input = self.normalize_title(user_input_title)
        # Get the list of normalized titles
        titles = self.df['normalized_title'].tolist()
        # Use RapidFuzz to find the best match
        matches = process.extractOne(normalized_input, titles, scorer=fuzz.token_set_ratio)
        if matches:
            best_match_title = matches[0]
            # Get the index of the best match
            index = self.df[self.df['normalized_title'] == best_match_title].index[0]
            original_title = self.df.loc[index, 'title']
            return index, original_title  # Return index and the original title
        else:
            # Movie not found
            return None, None

    # Recommendation method with optional filters and sorting
    # recommender/utils/recommendation_engine.py

    def recommend(self, movie_user_likes, num_recommendations=5, genres=None, sort_by='relevance'):
        movie_index, matched_title = self.get_index_from_title(movie_user_likes)

        if movie_index is not None:
            similar_movies = list(enumerate(self.cosine_sim[movie_index]))  # Access the row corresponding to the movie

            # Exclude the input movie itself
            similar_movies = similar_movies[1:]

            # Apply filters
            filtered_movies = self.apply_filters(similar_movies, genres)

            # Apply sorting
            sorted_movies = self.sort_movies(filtered_movies, sort_by)

            # Get top N recommendations
            recommendations = []
            i = 0
            for element in sorted_movies:
                recommended_movie = self.df.iloc[element[0]]
                recommended_title = recommended_movie['title']
                homepage = recommended_movie['homepage']
                movie_genres = recommended_movie['genres']
                release_date = recommended_movie['release_date']
                overview = recommended_movie['overview']
                recommendations.append({
                    'title': recommended_title,
                    'homepage': homepage,
                    'genres': movie_genres,
                    'release_date': release_date,
                    'overview': overview,
                })
                i += 1
                if i >= num_recommendations:
                    break
            return recommendations, matched_title
        else:
            # Movie not found
            return [], None

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
    def sort_movies(self, movies, sort_by):
        if sort_by == 'relevance':
            # Movies are already sorted by relevance (similarity score)
            return movies
        elif sort_by == 'popularity':
            movies = sorted(movies, key=lambda x: self.df.iloc[x[0]]['popularity'], reverse=True)
        elif sort_by == 'votes':
            movies = sorted(movies, key=lambda x: self.df.iloc[x[0]]['vote_count'], reverse=True)
        elif sort_by == 'title':
            movies = sorted(movies, key=lambda x: self.df.iloc[x[0]]['title'])
        return movies

    # Optional method to get genres of a movie
    def get_movie_genres(self, movie_title):
        index = self.df[self.df['title'] == movie_title].index[0]
        return self.df.loc[index, 'genres']
