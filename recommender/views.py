# recommender/views.py

from django.shortcuts import render
from .utils.recommendation_engine import MovieRecommender

# Initialize the recommender (do this once, outside of view functions)
recommender = MovieRecommender()

def index(request):
    context = {
        'genres': recommender.available_genres,
    }
    return render(request, 'index.html', context)

def results(request):
    movie_title = request.GET.get('movie_title', '')
    selected_genres = request.GET.getlist('genres')
    sort_by = request.GET.get('sort_by', 'relevance')
    num_recommendations = request.GET.get('num_recommendations', 5)

    # Convert number of recommendations to integer
    try:
        num_recommendations = int(num_recommendations)
        if num_recommendations < 1:
            num_recommendations = 5  # Default value if invalid
    except ValueError:
        num_recommendations = 5  # Default value if conversion fails

    # Convert selected genres to a set
    genres = set(selected_genres) if selected_genres else None

    if movie_title:
        # Set your desired threshold here
        threshold = 70
        recommendations, matched_title = recommender.recommend(
            movie_user_likes=movie_title,
            num_recommendations=num_recommendations,
            genres=genres,
            sort_by=sort_by,
            threshold=threshold  # Pass the threshold to the recommend method
        )
        if matched_title:
            if not recommendations:
                context = {
                    'error': 'No recommendations found with the applied filters.',
                    'matched_title': matched_title,
                    'genres': recommender.available_genres,
                }
                return render(request, 'index.html', context)
            context = {
                'recommendations': recommendations,
                'matched_title': matched_title,
            }
            return render(request, 'results.html', context)
        else:
            # Movie not found, handle accordingly
            return render(request, 'index.html', {'error': 'Movie not found. Please try another title.', 'genres': recommender.available_genres})
    else:
        return render(request, 'index.html', {'error': 'Please enter a movie title.', 'genres': recommender.available_genres})
