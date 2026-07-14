
import os
import requests
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from model_utils import (
    load_movie_data,
    preprocess_genres,
    build_embedding_matrix,
    recommend_movies,
)

load_dotenv()
TMDB_API_KEY = os.getenv("TMDB_API_KEY")

st.set_page_config(page_title="Find Your Next Movie", page_icon="🎬", layout="wide")

st.markdown("""
<style>
.movie-card{
background:#1e1e2f;border-radius:10px;padding:20px;margin-bottom:25px;
border:1px solid #3a3a52;min-height:180px;
box-shadow:2px 2px 12px rgba(0,0,0,.2);}
.movie-title{color:#ff4b4b;font-size:20px;font-weight:bold;}
.movie-meta{color:#a0a0b8;font-size:14px;margin-bottom:10px;}
.movie-overview{color:#e0e0e6;font-size:14px;}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    df = load_movie_data("movies.csv")
    return preprocess_genres(df)

@st.cache_resource
def build_similarity(df):
    return build_embedding_matrix(df)

def get_tmdb_movie_details(title):
    if not TMDB_API_KEY:
        return {"overview":"","rating":""}
    try:
        clean = title.split(" (")[0].strip()
        url="https://api.themoviedb.org/3/search/movie"
        params={"api_key":TMDB_API_KEY,"query":clean,"language":"en-US"}
        res=requests.get(url,params=params,timeout=5).json()
        if res.get("results"):
            m=res["results"][0]
            return {
                "overview":m.get("overview",""),
                "rating":f"{round(m.get('vote_average',0),1)} / 10" if m.get("vote_average",0)>0 else ""
            }
    except:
        pass
    return {"overview":"","rating":""}

def render_movie_grid(df):
    cols=4
    for i in range(0,len(df),cols):
        row=df.iloc[i:i+cols]
        c=st.columns(cols)
        for j,(_,movie) in enumerate(row.iterrows()):
            meta=get_tmdb_movie_details(movie["title"])
            with c[j]:
                st.markdown(f"""
                <div class="movie-card">
                <div class="movie-title">{movie['title']}</div>
                <div class="movie-meta">⭐ {meta['rating']}<br>{movie['genres'].replace('|',' • ')}</div>
                <div class="movie-overview">{meta['overview']}</div>
                </div>
                """,unsafe_allow_html=True)

def main():
    st.title("🎬 Find Your Next Movie")
    st.write("Discover movie recommendations based on genre similarity.")

    df=load_data()
    similarity=build_similarity(df)

    movie=st.sidebar.selectbox(
        "Select Movie",
        ["-- Select a Movie --"]+sorted(df["title"].unique().tolist())
    )

    genres=sorted({g for s in df["genres"] for g in s.split("|")})
    selected=st.sidebar.multiselect("Genre Filter",genres)
    top_k=st.sidebar.slider("Recommendations",4,16,8,step=4)

    if selected:
        mask=df["genres"].apply(lambda x: all(g in x.split("|") for g in selected))
        res=df[mask].head(top_k)
        st.subheader("Genre Results")
        render_movie_grid(res)

    elif movie!="-- Select a Movie --":
        recs=recommend_movies(df,similarity,movie,top_k)
        st.subheader(f"Recommendations for {movie}")
        render_movie_grid(recs)
    else:
        st.info("Select a movie or genre from the sidebar.")

if __name__=="__main__":
    main()
