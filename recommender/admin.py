from multiprocessing.resource_tracker import register

from django.contrib import admin

from .models import Movie, UserMovieLike


@admin.register(Movie)
class MovieAdmin(admin.ModelAdmin):
    pass
@admin.register(UserMovieLike)
class UserMovieLikeAdmin(admin.ModelAdmin):
    pass
# Register your models here.
