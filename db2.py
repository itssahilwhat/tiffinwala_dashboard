import streamlit as st
import pandas as pd
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from textblob import TextBlob
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import numpy as np
import json
import nltk

# Download NLTK stopwords
nltk.download('stopwords')
nltk.download('punkt')

# Setup Streamlit
st.set_page_config(page_title="Tiffinwala Analytics", layout="wide", page_icon="🥘")
plt.style.use('ggplot')

# Load data
@st.cache_data
def load_data():
    try:
        with open('tiffinwala_reviews.json') as f:
            data = json.load(f)
        df = pd.DataFrame(data)
        df['dates'] = pd.to_datetime(df['dates'])
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()  # Return empty DataFrame on failure

df = load_data()

# List of dish keywords
dish_keywords = ['dal', 'paneer', 'thepla', 'khandvi', 'undhiyu', 'kadhi', 'fafda', 'bajra rotla', 'Sev Tameta Nu Shaak',
                 'Ringan No Olo', 'Bhinda Nu Shaak', 'Dudhi Nu Shaak', 'Batata Nu Shaak', 'Lasaniya Batata', 'Karela Nu Shaak',
                 'Dudhi Chana Nu Shaak', 'Patra', 'Khaman', 'Handvo', 'Puri', 'Mooli Paratha', 'Masala Khichdi', 'Ghughra']

# Preprocess data
@st.cache_data
def enhance_data(df):
    if df.empty:
        return df  # Return empty DataFrame if no data

    # Sentiment analysis
    df['sentiment'] = df['description'].apply(lambda x: TextBlob(x).sentiment.polarity)
    df['sentiment_category'] = pd.cut(df['sentiment'],
                                      bins=[-1, -0.1, 0.1, 1],
                                      labels=['Negative', 'Neutral', 'Positive'])

    # Summarize reviews
    df['review_summary'] = df['description'].apply(lambda x: summarize_review(x))

    # Extract mentioned dishes
    for dish in dish_keywords:
        df[dish] = df['description'].str.contains(dish, case=False, na=False)

    return df

# Review Summarization using TextBlob
def summarize_review(text, num_sentences=3):
    blob = TextBlob(text)
    sentences = blob.sentences
    sorted_sentences = sorted(sentences, key=lambda s: len(s), reverse=True)
    summary = ' '.join([str(sorted_sentences[i]) for i in range(min(num_sentences, len(sorted_sentences)))])
    return summary

df = enhance_data(df)

# Train simple prediction model
@st.cache_resource
def train_model(df):
    if df.empty:
        return None, None  # Return None if no data

    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
    X = vectorizer.fit_transform(df['description'])
    y = (df['ratings'] <= 2).astype(int)  # Predict low ratings
    model = LogisticRegression()
    model.fit(X, y)
    return vectorizer, model

vectorizer, model = train_model(df)

# ======================
# Dashboard Layout
# ======================

st.title("🍱 Tiffinwala Excellence Dashboard")
st.markdown("""
**Hackathon-ready analytics** for Tiffinwala's customer feedback data  
*1000+ reviews · Real-time insights · Predictive analytics*
""")

# Sidebar Controls
st.sidebar.header("🔍 Filters")
if not df.empty:
    selected_ratings = st.sidebar.multiselect(
        "Select Ratings",
        options=sorted(df['ratings'].unique()),
        default=[1, 2, 3, 4, 5]
    )

    date_range = st.sidebar.date_input(
        "📅 Date Range",
        value=[df['dates'].min(), df['dates'].max()],
        min_value=df['dates'].min(),
        max_value=df['dates'].max()
    )

    # Apply filters
    filtered_df = df[
        (df['ratings'].isin(selected_ratings)) &
        (df['dates'] >= pd.to_datetime(date_range[0])) &
        (df['dates'] <= pd.to_datetime(date_range[1]))
        ]
else:
    filtered_df = pd.DataFrame()

# ======================
# Key Metrics
# ======================
st.header("📊 Performance Overview")

metric_cols = st.columns(4)

if not filtered_df.empty:
    metric_cols[0].metric("Total Reviews", len(filtered_df))
    metric_cols[1].metric("Avg Rating", f"{filtered_df['ratings'].mean():.1f} ⭐")
    positive_sentiment = f"{filtered_df[filtered_df['sentiment_category'] == 'Positive'].shape[0] / len(filtered_df):.0%}"
    metric_cols[2].metric("Positive Sentiment", positive_sentiment)
    low_rating_risk = f"{model.predict_proba(vectorizer.transform(filtered_df['description']))[:,1].mean():.0%}" if model else "N/A"
    metric_cols[3].metric("Low Rating Risk", low_rating_risk)
else:
    st.warning("No data available for the selected filters.")

# ======================
# Visualizations
# ======================
viz_col1, viz_col2 = st.columns(2)

with viz_col1:
    if not filtered_df.empty:
        st.subheader("📈 Rating Trends")
        time_df = filtered_df.set_index('dates').resample('W')['ratings'].mean().reset_index()
        fig = px.line(time_df, x='dates', y='ratings', template='plotly_white', height=300)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("😃😐😡 Sentiment Distribution")
    if not filtered_df.empty:
        sentiment_counts = filtered_df['sentiment_category'].value_counts()
        fig = px.pie(sentiment_counts, names=sentiment_counts.index, values=sentiment_counts.values, hole=0.4)
        st.plotly_chart(fig, use_container_width=True)

with viz_col2:
    st.subheader("🔍 Top Keywords")
    if not filtered_df.empty:
        text = ' '.join(filtered_df['description'].str.lower())
        if text.strip():
            stop_words = set(stopwords.words('english'))
            wordcloud = WordCloud(width=800, height=400, stopwords=stop_words).generate(text)
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            st.pyplot(plt, use_container_width=True)

    st.subheader("🍽️ Dish Mentions")
    if not filtered_df.empty:
        dish_counts = filtered_df[dish_keywords].sum().sort_values(ascending=False)
        fig = px.bar(dish_counts, orientation='h', labels={'value': 'Mentions', 'index': 'Dish'})
        st.plotly_chart(fig, use_container_width=True)

# ======================
# Review Summaries
# ======================
st.header("📋 Review Summaries")
if not filtered_df.empty:
    st.dataframe(filtered_df[['customer_name', 'ratings', 'dates', 'review_summary']].sort_values('dates', ascending=False),
                 hide_index=True)
else:
    st.warning("No reviews available.")

