# recommender/utils/recommendation_engine.py

import pandas as pd
import numpy as np
import nltk
import re
import warnings
import ssl

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import CountVectorizer
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
        self.features = ['keywords', 'cast', 'genres', 'director']

        # Handle missing values
        # Remove rows with missing values in selected features
        self.df = self.df.dropna(subset=self.features)

        # Apply text cleaning to each feature
        for feature in self.features:
            self.df[feature] = self.df[feature].apply(self.clean_text)

        # Standardize genres
        self.df['genres'] = self.df['genres'].apply(self.standardize_genres)

        # Combine features into a single string
        self.df['combined_features'] = self.df.apply(self.combine_features, axis=1)

        # Create CountVectorizer and compute count matrix
        self.cv = CountVectorizer()
        self.count_matrix = self.cv.fit_transform(self.df['combined_features'])

        # Compute cosine similarity matrix
        self.cosine_sim = cosine_similarity(self.count_matrix)

        # Normalize movie titles by removing articles and lowercasing
        self.df['normalized_title'] = self.df['title'].apply(self.normalize_title)

    # Function to clean text data
    def clean_text(self, text):
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

    # Standardize genres
    def standardize_genres(self, text):
        # Replace 'sci fi' with 'science fiction'
        text = text.replace('sci fi', 'science fiction')
        # Add other replacements as needed
        return text

    # Combine features into a single string
    def combine_features(self, row):
        return ' '.join([row['keywords'], row['cast'], row['genres'], row['director']])

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

    # Recommendation method
    def recommend(self, movie_user_likes, num_recommendations=5):
        movie_index, matched_title = self.get_index_from_title(movie_user_likes)

        if movie_index is not None:
            similar_movies = list(enumerate(self.cosine_sim[movie_index]))  # Access the row corresponding to the movie

            # Sort movies based on similarity scores
            sorted_similar_movies = sorted(similar_movies, key=lambda x: x[1], reverse=True)[1:]

            recommendations = []
            i = 0
            for element in sorted_similar_movies:
                recommended_title = self.get_title_from_index(element[0])
                recommendations.append(recommended_title)
                i += 1
                if i >= num_recommendations:
                    break
            return recommendations, matched_title
        else:
            # Movie not found
            return [], None

    # Optional method to get genres of a movie
    def get_movie_genres(self, movie_title):
        index = self.df[self.df['title'] == movie_title].index[0]
        return self.df.loc[index, 'genres']
