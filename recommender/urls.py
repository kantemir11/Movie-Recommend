from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('results/', views.results, name='results'),
    path('signup/', views.signup, name='signup'),
    path('like_movie/', views.like_movie, name='like_movie'),

]
