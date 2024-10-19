# recommender/views.py

from django.shortcuts import render
from .utils.recommendation_engine import MovieRecommender

# Initialize the recommender (do this once, outside of view functions)
recommender = MovieRecommender()

def index(request):
    return render(request, 'index.html')

def results(request):
    movie_title = request.GET.get('movie_title', '')
    if movie_title:
        recommendations, matched_title = recommender.recommend(movie_title)
        if matched_title:
            context = {
                'recommendations': recommendations,
                'matched_title': matched_title,
            }
            return render(request, 'results.html', context)
        else:
            # Movie not found, handle accordingly
            return render(request, 'index.html', {'error': 'Movie not found. Please try another title.'})
    else:
        return render(request, 'index.html', {'error': 'Please enter a movie title.'})
