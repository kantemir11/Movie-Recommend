Personalized Movie Recommendation System

A web-based application that provides personalized movie recommendations to users based on their liked movies. Built with Python and Django, the system allows users to search for movies, like them, and receive tailored recommendations. The application features user authentication, interactive UI elements, and advanced sorting options to enhance user engagement.

DEMO: https://drive.google.com/file/d/1PUsxjBjvJXy0LRmfCyLm2XvmW2wqAo5-/view?usp=sharing

Table of Contents

	•	Features
	•	Technologies Used
	•	Installation
	•	Prerequisites
	•	Setup Instructions
	•	Usage
	•	Screenshots
	•	Contributing
	•	License

Features

	•	User Authentication: Secure user registration and login system.
	•	Movie Search: Users can search for movies by title.
	•	Like Movies: Users can like or unlike movies, with an interactive animated button.
	•	Personalized Recommendations: Recommendations based on the user’s liked movies.
	•	Sorting Options: Sort recommendations by relevance, popularity, rating, release date, or title.
	•	Genre Filtering: Filter recommendations by selected genres.
	•	Responsive Design: User-friendly interface that works across devices.
	•	Interactive UI Elements: Animated Like button with visual feedback and confetti effect.
	•	Error Handling: Informative messages for empty inputs or unavailable movies.
	•	Data Visualization: Display movie details including rating, popularity, genres, and overview.

Technologies Used

	•	Python 
	•	Django 
	•	Pandas
	•	NumPy
	•	scikit-learn
	•	Bootstrap 4
	•	jQuery
	•	Font Awesome
	•	Select2
	•	canvas-confetti

Installation

Prerequisites

	•	Python3 installed on your system.
	•	pip package installer.
	•	Git (optional, for cloning the repository).

Setup Instructions

	1.	Clone the Repository

git clone https://github.com/yourusername/movie-recommendation-system.git
cd movie-recommendation-system


	2.	Create a Virtual Environment
It’s recommended to use a virtual environment to manage dependencies.

python -m venv venv


	3.	Activate the Virtual Environment
	•	On Windows:

venv\Scripts\activate


	•	On macOS and Linux:

source venv/bin/activate


	4.	Install Required Packages

pip install -r requirements.txt


	5.	Set Up the Database
Apply migrations to set up the SQLite database.

python manage.py migrate


	6.	Load Movie Data
Ensure you have a CSV file containing movie data. Place it in the appropriate directory and update the path in recommendation_engine.py.

# In recommender/utils/recommendation_engine.py
self.df = pd.read_csv('path_to_your_movies.csv')


	7.	Collect Static Files

python manage.py collectstatic


	8.	Create a Superuser (Optional)

python manage.py createsuperuser


	9.	Run the Development Server

python manage.py runserver



Usage

	1.	Access the Application
Open your web browser and navigate to http://127.0.0.1:8000/.
	2.	Register a New Account
	•	Click on the Sign Up link in the navigation bar.
	•	Fill out the registration form and submit.
	3.	Log In
	•	After registering, log in using your new credentials.
	4.	Search for Movies
	•	On the homepage, enter a movie title in the search bar.
	•	Optionally, adjust the number of recommendations, select genres, or choose sorting options.
	•	Click Get Recommendations.
	5.	Like Movies
	•	In the results, click the Like button next to movies you enjoy.
	•	The button will animate and change to Liked.
	6.	View Liked Movies
	•	Return to the homepage to see your list of liked movies.
	•	You can unlike movies directly from this list.
	7.	Receive Personalized Recommendations
	•	Based on your liked movies, the system will provide tailored recommendations.

Screenshots

The home page with search options and liked movies.

Search results with options to like movies.

User’s liked movies displayed on the home page.

Contributing

Contributions are welcome! If you’d like to contribute to this project:

	1.	Fork the repository.
	2.	Create a new branch: git checkout -b feature-name.
	3.	Make your changes and commit them: git commit -m 'Add new feature'.
	4.	Push to the branch: git push origin feature-name.
	5.	Submit a pull request.

License

This project is licensed under the MIT License - see the LICENSE file for details.

Contact

For any inquiries or feedback, please contact:

	•	Your Name
	•	Email: your.email@example.com
	•	GitHub: yourusername

Note: Replace placeholders like yourusername, your.email@example.com, and path_to_your_movies.csv with your actual information.

Additional Notes

	•	Data Source: Ensure you have the appropriate rights to use the movie data in your CSV file.
	•	Security: Do not expose your secret keys or sensitive information in a public repository.
	•	Deployment: For production deployment, consider using a robust database like PostgreSQL and configure appropriate security measures.

Feel free to customize this README file further to suit your project’s specifics. Include any additional information that might be helpful for users or contributors.
