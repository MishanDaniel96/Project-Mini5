import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

st.set_page_config(page_title="ChatGPT Sentiment Dashboard", layout="wide")

@st.cache_data
def load_data():
    return pd.read_csv('C:/Users/danie/Downloads/project/guvi/Project mini 5/cleaned_reviews.csv')

df = load_data()

st.title("📊 ChatGPT User Sentiment Analysis")
st.markdown("Insights into customer satisfaction and application performance.")


st.sidebar.header("Filters")
platform_filter = st.sidebar.multiselect("Platform:", df['platform'].unique(), default=df['platform'].unique())
version_filter = st.sidebar.multiselect("Version:", df['version'].unique(), default=df['version'].unique()[:5])

filtered_df = df[(df['platform'].isin(platform_filter)) & (df['version'].isin(version_filter))]

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Reviews", len(filtered_df))
col2.metric("Avg Rating", f"{filtered_df['rating'].mean():.2f} ⭐")
col3.metric("Positive %", f"{(filtered_df['sentiment']=='Positive').mean()*100:.1f}%")
col4.metric("Negative %", f"{(filtered_df['sentiment']=='Negative').mean()*100:.1f}%")

tab1, tab2, tab3 = st.tabs(["Sentiment Trends", "Keywords", "Platform & Versions"])

with tab1:
    st.subheader("Overall Sentiment Distribution")
    fig1, ax1 = plt.subplots()
    sns.countplot(data=filtered_df, x='sentiment', palette='viridis', ax=ax1)
    st.pyplot(fig1)
    
    st.subheader("Rating Distribution (1-5)")
    fig2, ax2 = plt.subplots()
    sns.histplot(filtered_df['rating'], bins=5, kde=False, color='skyblue', ax=ax2)
    st.pyplot(fig2)

with tab2:
    col_a, col_b = st.columns(2)
    with col_a:
        st.write("### Positive Keywords")
        pos_text = " ".join(filtered_df[filtered_df['sentiment']=='Positive']['cleaned_review'])
        wc_pos = WordCloud(background_color="white").generate(pos_text)
        st.image(wc_pos.to_array())
    with col_b:
        st.write("### Negative Keywords")
        neg_text = " ".join(filtered_df[filtered_df['sentiment']=='Negative']['cleaned_review'])
        wc_neg = WordCloud(background_color="white").generate(neg_text)
        st.image(wc_neg.to_array())

with tab3:
    st.write("### Avg Rating by Platform")
    platform_avg = filtered_df.groupby('platform')['rating'].mean().sort_values()
    st.bar_chart(platform_avg)
    
    st.write("### Avg Rating by Version")
    version_avg = filtered_df.groupby('version')['rating'].mean().sort_values(ascending=False)
    st.bar_chart(version_avg)

st.subheader("🔍 Explore Reviews")
search_query = st.text_input("Search reviews by keyword:")
if search_query:
    st.write(filtered_df[filtered_df['review'].str.contains(search_query, case=False)][['username', 'rating', 'sentiment', 'review']])
else:
    st.write(filtered_df[['username', 'rating', 'sentiment', 'review']].head(10))