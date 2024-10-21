# recommender/models.py

from django.db import models
from django.contrib.auth.models import User

class Movie(models.Model):
    movie_id = models.IntegerField(unique=True)
    title = models.CharField(max_length=255)
    # Add other fields as needed (e.g., genres, release_date)

    def __str__(self):
        return self.title

class UserMovieLike(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    movie = models.ForeignKey(Movie, on_delete=models.CASCADE)
    liked_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ('user', 'movie')

    def __str__(self):
        return f"{self.user.username} likes {self.movie.title}"
