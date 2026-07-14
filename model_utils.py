
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def load_movie_data(path="movies.csv"):
    df = pd.read_csv(path)
    return df

def preprocess_genres(df):
    df = df.copy()
    df["genres"] = df["genres"].fillna("")
    return df

def build_embedding_matrix(df):
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf = vectorizer.fit_transform(df["genres"].str.replace("|", " ", regex=False))
    return cosine_similarity(tfidf)

def recommend_movies(df, similarity_matrix, movie_title, top_k=8):
    if movie_title not in df["title"].values:
        return pd.DataFrame()

    idx = df[df["title"] == movie_title].index[0]
    scores = list(enumerate(similarity_matrix[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)

    movie_indices = [i[0] for i in scores[1:top_k+1]]
    return df.iloc[movie_indices]
