from fastapi import FastAPI, Query
from typing import List, Optional
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

app = FastAPI()

# Read data
df = pd.read_csv("./netflix_titles.csv")

# dropping the rows having NaN values
df.dropna(subset=["director", "title", "description"], inplace=True)

# To reset the indices
df = df.reset_index(drop=True)

# TF-IDF Vectorization for each relevant text column
tfidf_vectorizer_title = TfidfVectorizer(stop_words="english")
tfidf_matrix_title = tfidf_vectorizer_title.fit_transform(df["title"])

tfidf_vectorizer_director = TfidfVectorizer(stop_words="english")
tfidf_matrix_director = tfidf_vectorizer_director.fit_transform(df["director"])

tfidf_vectorizer_description = TfidfVectorizer(stop_words="english")
tfidf_matrix_description = tfidf_vectorizer_description.fit_transform(df["description"])


# Function to get the top N matching records for each search
def search_by_title(query: str, n: int = 5) -> List[dict]:
    search_vector_title = tfidf_vectorizer_title.transform([query])
    cosine_similarities_title = linear_kernel(
        search_vector_title, tfidf_matrix_title
    ).flatten()
    most_matching_indices_title = cosine_similarities_title.argsort()[: -n - 1 : -1]
    results = df.loc[most_matching_indices_title].to_dict(orient="records")
    for i, idx in enumerate(most_matching_indices_title):
        results[i]["similarity_score"] = cosine_similarities_title[idx]
    return results


def search_by_director(query: str, n: int = 5) -> List[dict]:
    search_vector_director = tfidf_vectorizer_director.transform([query])
    cosine_similarities_director = linear_kernel(
        search_vector_director, tfidf_matrix_director
    ).flatten()
    most_matching_indices_director = cosine_similarities_director.argsort()[
        : -n - 1 : -1
    ]
    results = df.loc[most_matching_indices_director].to_dict(orient="records")
    for i, idx in enumerate(most_matching_indices_director):
        results[i]["similarity_score"] = cosine_similarities_director[idx]
    return results


def search_by_description(query: str, n: int = 5) -> List[dict]:
    search_vector_description = tfidf_vectorizer_description.transform([query])
    cosine_similarities_description = linear_kernel(
        search_vector_description, tfidf_matrix_description
    ).flatten()
    most_matching_indices_description = cosine_similarities_description.argsort()[
        : -n - 1 : -1
    ]
    results = df.loc[most_matching_indices_description].to_dict(orient="records")
    for i, idx in enumerate(most_matching_indices_description):
        results[i]["similarity_score"] = cosine_similarities_description[idx]
    return results


# FastAPI Endpoints
@app.get("/search/title/{title}", response_model=List[dict])
async def search_title(title: str, n: Optional[int] = Query(5, ge=1, le=10)):
    return search_by_title(title, n)


@app.get("/search/director/{director}", response_model=List[dict])
async def search_director(director: str, n: Optional[int] = Query(5, ge=1, le=10)):
    return search_by_director(director, n)


@app.get("/search/description/{description}", response_model=List[dict])
async def search_description(
    description: str, n: Optional[int] = Query(5, ge=1, le=10)
):
    return search_by_description(description, n)
