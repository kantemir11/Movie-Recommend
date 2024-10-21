# recommender/views.py

from django.shortcuts import render, redirect
from django.contrib.auth import login, authenticate
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
from django.views.decorators.http import require_POST
from .utils.recommendation_engine import MovieRecommender
from .forms import SignUpForm
from .models import Movie, UserMovieLike

# Initialize the recommender (do this once, outside of view functions)
recommender = MovieRecommender()

def index(request):
    if request.user.is_authenticated:
        user_liked_movie_ids = UserMovieLike.objects.filter(user=request.user).values_list('movie__movie_id', flat=True)
        # Get details of liked movies
        liked_movies = recommender.get_movies_by_ids(user_liked_movie_ids)
        context = {
            'genres': recommender.available_genres,
            'liked_movies': liked_movies,
            'user_liked_movie_ids': list(user_liked_movie_ids),
        }
    else:
        context = {
            'genres': recommender.available_genres,
            'liked_movies': [],  # No liked movies for anonymous users
        }
    return render(request, 'index.html', context)

def results(request):
    movie_title = request.GET.get('movie_title', '').strip()
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

    if not movie_title:
        # If no movie title is provided, redirect back to the index page with an error
        return render(request, 'index.html', {
            'error': 'Please enter a movie title.',
            'genres': recommender.available_genres
        })

    # Proceed with the recommendation logic
    recommendations, matched_title = recommender.recommend(
        movie_user_likes=movie_title,
        num_recommendations=num_recommendations,
        genres=genres,
        sort_by=sort_by
    )

    if matched_title:
        if not recommendations:
            context = {
                'error': 'No recommendations found with the applied filters.',
                'matched_title': matched_title,
                'genres': recommender.available_genres,
            }
            return render(request, 'index.html', context)
        # Get user's liked movies
        if request.user.is_authenticated:
            user_liked_movie_ids = UserMovieLike.objects.filter(user=request.user).values_list('movie__movie_id', flat=True)
        else:
            user_liked_movie_ids = []
        context = {
            'recommendations': recommendations,
            'matched_title': matched_title,
            'user_liked_movie_ids': list(user_liked_movie_ids),
        }
        return render(request, 'results.html', context)
    else:
        # Movie not found, handle accordingly
        return render(request, 'index.html', {
            'error': 'Movie not found. Please check the title and try again.',
            'genres': recommender.available_genres
        })

def signup(request):
    if request.method == 'POST':
        form = SignUpForm(request.POST)
        if form.is_valid():
            user = form.save()
            # Automatically log in the user after registration
            username = form.cleaned_data.get('username')
            raw_password = form.cleaned_data.get('password1')
            user = authenticate(username=username, password=raw_password)
            login(request, user)
            return redirect('index')
    else:
        form = SignUpForm()
    return render(request, 'registration/signup.html', {'form': form})

@require_POST
@login_required
def like_movie(request):
    movie_id = request.POST.get('movie_id')
    if movie_id:
        try:
            movie_id = int(movie_id)
        except ValueError:
            return JsonResponse({'status': 'error', 'message': 'Invalid movie ID'})

        # Get or create the movie instance
        movie, created = Movie.objects.get_or_create(movie_id=movie_id, defaults={
            'title': request.POST.get('title', 'Unknown Title'),
            # Add other fields as needed
        })

        # Check if the user already liked this movie
        liked_movie = UserMovieLike.objects.filter(user=request.user, movie=movie).first()
        if liked_movie:
            # User already liked this movie; remove the like
            liked_movie.delete()
            return JsonResponse({'status': 'unliked'})
        else:
            # Add the like
            UserMovieLike.objects.create(user=request.user, movie=movie)
            return JsonResponse({'status': 'liked'})
    return JsonResponse({'status': 'error', 'message': 'Movie ID not provided'})
