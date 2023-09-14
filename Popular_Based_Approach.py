import streamlit as st
import pandas as pd
from IPython.display import Image, display

# Read the CSV file
df = pd.read_csv("books.csv")

# Set Streamlit options
st.set_option("deprecation.showfileUploaderEncoding", False)
st.set_option("deprecation.showPyplotGlobalUse", False)

# Sidebar to let user select the number of recommendations and quantile
st.sidebar.header("Recommendation Settings")
num_recommendations = st.sidebar.number_input(
    "Enter the number of books to recommend", min_value=1
)
quantile_value = st.sidebar.slider(
    "Select the quantile for minimum rating count (0 - 1)", 0.0, 1.0, 0.75, 0.01
)

# Add a button to trigger recommendations
recommend_button = st.sidebar.button("Get Recommendations")

# Extract relevant features and remove null values
books_data = df[
    [
        "authors",
        "original_publication_year",
        "title",
        "average_rating",
        "ratings_count",
        "image_url",
    ]
].copy()
books_data.dropna(inplace=True)

# Change data type of original_publication_year to int from float
books_data["original_publication_year"] = books_data[
    "original_publication_year"
].astype(int)

# Calculate mean of average_rating column
C = books_data["average_rating"].mean()

# Calculate the minimum number of ratings_count required to be in the chart, m
m = books_data["ratings_count"].quantile(quantile_value)


# Define weighted_rating function
def weighted_rating(x, m=m, C=C):
    v = x["ratings_count"]
    R = x["average_rating"]
    return (v / (v + m) * R) + (m / (m + v) * C)


# Define a function for popularity-based recommendations
def popularity_based_recommendations(n):
    recommendations = new_books_data.head(n)
    return recommendations


# Display recommendations when the button is clicked
if recommend_button:
    # Extract all qualified books into a new DataFrame
    new_books_data = books_data.copy().loc[books_data["ratings_count"] >= m]

    # Calculate weighted_rating for qualified books
    new_books_data["weighted_rating"] = new_books_data.apply(weighted_rating, axis=1)

    # Sort books by weighted_rating in descending order
    new_books_data = new_books_data.sort_values(by="weighted_rating", ascending=False)

    st.title("Popularity-Based Book Recommendations")
    st.write("These recommendations are based on popularity and weighted rating.")

    recommended_books = popularity_based_recommendations(num_recommendations)
    for i, (author, year, title, image_url) in enumerate(
        recommended_books[
            ["authors", "original_publication_year", "title", "image_url"]
        ].values,
        start=1,
    ):
        st.write(f"{i}. {title} ({year}) by {author}")
        st.image(image_url, caption=title, width=200, use_column_width=False)
        st.write("")
