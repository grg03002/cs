import dash
from dash import dcc, html, Input, Output, State
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
import os
from flask import send_from_directory

app = dash.Dash(__name__, title="Movie Recommendation App")
server = app.server

IMAGE_DIR = os.path.join(os.getcwd(), "MovieImages")

@app.server.route("/MovieImages/<path:image_name>")
def serve_image(image_name):
    return send_from_directory(IMAGE_DIR, image_name)

movies_file = "movies.dat"
ratings_file = "ratings.dat"

movies = pd.read_csv(
    movies_file, sep=",", header=None, names=["MovieID", "Title", "Genres"], encoding="latin1", on_bad_lines="skip"
)
ratings = pd.read_csv(
    ratings_file, sep=",", header=None, names=["UserID", "MovieID", "Rating", "Timestamp"], encoding="latin1"
)

rating_matrix = ratings.pivot(index="UserID", columns="MovieID", values="Rating")
row_means = rating_matrix.mean(axis=1, skipna=True)
normalized_matrix = rating_matrix.subtract(row_means, axis=0)

sparse_matrix = csr_matrix(normalized_matrix.fillna(0).values)
movie_similarity = cosine_similarity(sparse_matrix.T)
movie_similarity = (1 + movie_similarity) / 2  
movie_similarity_df = pd.DataFrame(
    movie_similarity, index=rating_matrix.columns, columns=rating_matrix.columns
)

pruned_similarity = movie_similarity_df.apply(lambda x: x.nlargest(30), axis=1)

def myIBCF(newuser, similarity_matrix, ratings_matrix):
    predicted_ratings = pd.Series(index=ratings_matrix.columns, dtype=float)
    for movie in ratings_matrix.columns:
        if movie not in similarity_matrix.columns:
            continue
        sim_scores = similarity_matrix[movie].dropna()
        rated_movies = newuser.dropna()
        intersecting_movies = sim_scores.index.intersection(rated_movies.index)
        if len(intersecting_movies) > 0:
            weights = sim_scores[intersecting_movies]
            ratings = rated_movies[intersecting_movies]
            predicted_ratings[movie] = (weights * ratings).sum() / weights.sum()
    return predicted_ratings.nlargest(10)

sample_movies = movies.sample(20)[["MovieID", "Title", "Genres"]]

app.layout = html.Div([
    html.H1("Movie Recommendation App"),
    html.H2("Rate These Movies"),
    html.Div(
        id="movie-ratings",
        children=[
            html.Div(
                [
                    html.Div(
                        [
                            html.Img(
                                src=f"/MovieImages/{row['MovieID']}.jpg",
                                style={"height": "100px", "margin-right": "20px"},
                            ),
                            html.Div(
                                [
                                    html.H4(f"{row['Title']} ({row['Genres']})"),
                                    dcc.RadioItems(
                                        id=f"rating-{row['MovieID']}",
                                        options=[{"label": str(i), "value": i} for i in range(1, 6)],
                                        labelStyle={"display": "inline-block", "margin-right": "10px"},
                                        style={"margin-top": "10px"}
                                    ),
                                ]
                            ),
                        ],
                        style={"display": "flex", "align-items": "center", "margin-bottom": "20px"},
                    )
                ]
            )
            for _, row in sample_movies.iterrows()
        ]
    ),
    html.Button("Submit Ratings", id="submit-button", n_clicks=0),
    html.H2("Recommendations"),
    html.Div(id="recommendations-output")
])

@app.callback(
    Output("recommendations-output", "children"),
    Input("submit-button", "n_clicks"),
    [State(f"rating-{row['MovieID']}", "value") for _, row in sample_movies.iterrows()]
)
def generate_recommendations(n_clicks, *ratings):
    if n_clicks == 0:
        return "Submit for recommendations."

    user_ratings = {movie_id: rating for movie_id, rating in zip(sample_movies["MovieID"], ratings) if rating}
    if not user_ratings:
        return "Please rate at least one movie."

    newuser = pd.Series(index=rating_matrix.columns, dtype=float)
    for movie_id, rating in user_ratings.items():
        newuser[movie_id] = rating

    recommendations = myIBCF(newuser, pruned_similarity, rating_matrix)

    recs = []
    for movie_id, score in recommendations.items():
        movie_row = movies[movies["MovieID"] == movie_id]
        if not movie_row.empty:
            title = movie_row.iloc[0]["Title"]
            recs.append(html.Div([
                html.Img(
                    src=f"/MovieImages/{movie_id}.jpg",
                    style={"height": "100px", "margin-right": "20px"},
                ),
                html.Div(f"{title}: Predicted Rating {score:.2f}")
            ], style={"display": "flex", "align-items": "center", "margin-bottom": "10px"}))
        else:
            recs.append(html.Div(f"Movie ID {movie_id} not found: Predicted Rating {score:.2f}"))

    return recs

if __name__ == "__main__":
    app.run_server(debug=True)