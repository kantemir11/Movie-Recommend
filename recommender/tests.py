from utils.recommendation_engine import MovieRecommender

if __name__ == '__main__':
    recommender = MovieRecommender()
    movie_title = 'Inception'
    recommendations, matched_title = recommender.recommend(movie_title)
    if matched_title:
        print(f"Top recommendations for '{matched_title}':")
        for movie in recommendations:
            print(movie)
    else:
        print("Movie not found. Please try another title.")
