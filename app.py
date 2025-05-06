# app.py (Streamlit entrypoint)
import streamlit as st
import pandas as pd
import zipfile
import io
from explorer.data import load_corpus, compute_term_freq
from explorer.viz import plot_term_frequency, plot_wordcloud


def main():
    st.title("Topic Explorer Dashboard")
    st.write("Upload a CSV/JSON file or a ZIP of .txt files to explore term frequencies.")

    uploaded = st.file_uploader(
        "Upload corpus:", type=["csv", "json", "zip"]
    )
    if uploaded:
        try:
            df = load_corpus(uploaded)
        except Exception as e:
            st.error(f"Failed to load corpus: {e}")
            return

        st.sidebar.header("Settings")
        top_n = st.sidebar.slider("Top N Terms", min_value=5, max_value=50, value=20)

        freq_df = compute_term_freq(df, top_n=top_n)

        st.header("Term Frequency Bar Chart")
        fig1 = plot_term_frequency(freq_df, top_n)
        st.pyplot(fig1)

        st.header("Word Cloud")
        fig2 = plot_wordcloud(freq_df)
        st.pyplot(fig2)

        if 'source' in df.columns:
            st.header("Sources")
            st.table(df[['source']].drop_duplicates())

if __name__ == "__main__":
    main()
