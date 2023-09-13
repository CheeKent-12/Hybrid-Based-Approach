import streamlit as st
import pandas as pd
from prettytable import PrettyTable
from IPython.display import Image, display
from surprise import Reader, Dataset, SVD, KNNBasic, NMF
from surprise.model_selection import train_test_split


# Load and preprocess your dataset (Assuming you have 'ratings.csv' and 'books.csv')
def load_and_preprocess_data():
    ratings = pd.read_csv("ratings.csv")
    books = pd.read_csv("books.csv")

    # Drop rows with missing values
    books.dropna(inplace=True)

    return ratings, books


ratings, books = load_and_preprocess_data()


# Function to recommend books based on the selected model
def recommend_books(model, userId, book_title):
    # Create a user-item rating matrix (Collaborative Filtering)
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(ratings[["user_id", "book_id", "rating"]], reader)
    trainset, _ = train_test_split(data, test_size=0.25)

    if model == "SVD":
        recommender = SVD()
    elif model == "KNN":
        recommender = KNNBasic(
            k=40, sim_options={"name": "cosine", "user_based": False}
        )
    elif model == "NMF":
        recommender = NMF()
    else:
        raise ValueError("Invalid model type")

    recommender.fit(trainset)

    # Find the book_id for the given book title
    book_id = books[books["title"] == book_title]["book_id"].values[0]

    # Get book recommendations
    if model == "SVD":
        predictions = [
            (book_id, recommender.predict(userId, book_id).est)
            for book_id in books["book_id"]
        ]
    else:
        predictions = [
            (book_id, recommender.predict(userId, book_id).est)
            for book_id in books["book_id"]
        ]

    # Sort books by predicted ratings in descending order
    recommended_books = books.copy()
    recommended_books["est"] = [
        est for _, est in sorted(predictions, key=lambda x: x[1], reverse=True)
    ]
    recommended_books = recommended_books.sort_values("est", ascending=False)

    # Truncate or wrap long titles here (e.g., to 50 characters)
    recommended_books["title"] = recommended_books["title"].apply(
        lambda x: x[:50] + "..." if len(x) > 50 else x
    )

    return recommended_books


# Streamlit UI
st.title("Book Recommendation App")

# User input for user ID and book title
user_id = st.sidebar.number_input("Enter User ID:", min_value=1)
book_title = st.sidebar.text_input("Enter Book Title:")

# Display the number input widget for the number of recommendations
num_recommendations = st.sidebar.number_input(
    "Enter the number of books for recommendation:",
    min_value=1,
    # value=5  # Set a default value
)

# Sidebar to choose the recommendation model
model_choice = st.sidebar.radio("Select a Recommendation Model:", ("SVD", "KNN", "NMF"))


if st.sidebar.button("Get Recommendations"):
    if book_title not in books["title"].values:
        st.sidebar.error(f"Book title '{book_title}' not found in the dataset.")
    else:
        recommended_books = recommend_books(model_choice, user_id, book_title)

        # Display book recommendations in a table
        st.write(f"Recommendations using {model_choice}:")
        table = PrettyTable()
        table.field_names = [
            "Title",
            "Authors",
            "Year",
            "Average Rating",
            "Ratings Count",
        ]

        for _, row in recommended_books.head(num_recommendations).iterrows():
            table.add_row(
                [
                    row["title"],
                    row["authors"],
                    int(row["original_publication_year"]),
                    row["average_rating"],
                    row["ratings_count"],
                ]
            )

        st.write(table)

        # Display book images for the recommended books
        for _, row in recommended_books.head(num_recommendations).iterrows():
            st.image(row["image_url"], caption=row["title"])
