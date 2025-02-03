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

# Setup
st.set_page_config(page_title="Tiffinwala Analytics", layout="wide", page_icon="🥘")
plt.style.use('ggplot')

# Load data
@st.cache_data
def load_data():
    with open('tiffinwala_reviews.json') as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    df['dates'] = pd.to_datetime(df['dates'])
    return df

df = load_data()

# List of dish keywords
dish_keywords = ['dal', 'paneer', 'thepla', 'khandvi', 'undhiyu', 'kadhi', 'fafda', 'bajra rotla', 'Sev Tameta Nu Shaak',
                 'Ringan No Olo', 'Bhinda Nu Shaak', 'Dudhi Nu Shaak', 'Batata Nu Shaak', 'Lasaniya Batata', 'Karela Nu Shaak',
                 'Dudhi Chana Nu Shaak', 'Patra', 'Khaman', 'Handvo', 'Puri', 'Mooli Paratha', 'Masala Khichdi', 'Ghughra']  # Ensure this is defined globally

# Preprocess data
@st.cache_data
def enhance_data(df):
    # Sentiment analysis
    df['sentiment'] = df['description'].apply(lambda x: TextBlob(x).sentiment.polarity)
    df['sentiment_category'] = pd.cut(df['sentiment'],
                                      bins=[-1, -0.1, 0.1, 1],
                                      labels=['Negative', 'Neutral', 'Positive'])

    # Summarize reviews
    df['review_summary'] = df['description'].apply(lambda x: summarize_review(x))

    # Extract mentioned dishes
    for dish in dish_keywords:
        df[dish] = df['description'].str.contains(dish, case=False)

    return df

# Review Summarization using TextBlob
def summarize_review(text, num_sentences=3):
    blob = TextBlob(text)
    sentences = blob.sentences
    sorted_sentences = sorted(sentences, key=lambda s: len(s), reverse=True)
    summary = ' '.join([str(sorted_sentences[i]) for i in range(min(num_sentences, len(sorted_sentences)))])
    return summary

# Generate a summary paragraph from all reviews
def generate_summary_review(df):
    positive_reviews = df[df['sentiment_category'] == 'Positive']
    negative_reviews = df[df['sentiment_category'] == 'Negative']

    positive_reviews_text = " ".join(positive_reviews['description'].head(10))  # Use top 10 positive reviews
    negative_reviews_text = " ".join(negative_reviews['description'].head(10))  # Use top 10 negative reviews

    positive_blob = TextBlob(positive_reviews_text)
    negative_blob = TextBlob(negative_reviews_text)

    # AI-generated summary from positive and negative reviews
    positive_summary = positive_blob.sentences[0] if len(positive_blob.sentences) > 0 else "Customers generally have positive feedback."
    negative_summary = negative_blob.sentences[0] if len(negative_blob.sentences) > 0 else "Negative feedback suggests improvements are needed."

    # Combine both into a final summary paragraph
    final_summary = f"Customers generally appreciate the variety and quality of dishes offered by Tiffinwala. {positive_summary}. On the other hand, {negative_summary} This combination of experiences provides a holistic view of customer satisfaction."
    return final_summary

df = enhance_data(df)

# Train simple prediction model
@st.cache_resource
def train_model(df):
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
st.markdown(""" **Hackathon-ready analytics** for Tiffinwala's customer feedback data  
*1000+ reviews · Real-time insights · Predictive analytics* """)

# Sidebar Controls
st.sidebar.header("🔍 Filters")
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

# ======================
# Key Metrics
# ======================
st.header("📊 Performance Overview")

metric_cols = st.columns(4)

# Ensure filtered_df is not empty before performing calculations
if len(filtered_df) > 0:
    metric_cols[0].metric("Total Reviews", len(filtered_df))
    metric_cols[1].metric("Avg Rating", f"{filtered_df['ratings'].mean():.1f} ⭐")
    positive_sentiment = f"{filtered_df[filtered_df['sentiment_category'] == 'Positive'].shape[0] / len(filtered_df):.0%}"
    metric_cols[2].metric("Positive Sentiment", positive_sentiment)
    low_rating_risk = f"{model.predict_proba(vectorizer.transform(filtered_df['description']))[:,1].mean():.0%}"
    metric_cols[3].metric("Low Rating Risk", low_rating_risk)
else:
    metric_cols[0].metric("Total Reviews", 0)
    metric_cols[1].metric("Avg Rating", "N/A")
    metric_cols[2].metric("Positive Sentiment", "N/A")
    metric_cols[3].metric("Low Rating Risk", "N/A")

# ======================
# Visualizations
# ======================
viz_col1, viz_col2 = st.columns(2)

with viz_col1:
    # Time Series Analysis
    st.subheader("📈 Rating Trends")
    time_df = filtered_df.set_index('dates').resample('W')['ratings'].mean().reset_index()
    fig = px.line(time_df, x='dates', y='ratings',
                  template='plotly_white',
                  labels={'ratings': 'Average Rating'},
                  height=300)
    fig.update_layout(margin=dict(l=0, r=0))
    st.plotly_chart(fig, use_container_width=True)

    # Sentiment Analysis
    st.subheader("😃😐😡 Sentiment Distribution")
    sentiment_counts = filtered_df['sentiment_category'].value_counts()
    fig = px.pie(sentiment_counts,
                 names=sentiment_counts.index,
                 values=sentiment_counts.values,
                 hole=0.4,
                 color_discrete_sequence=px.colors.qualitative.Pastel)
    st.plotly_chart(fig, use_container_width=True)

with viz_col2:
    # Word Cloud
    st.subheader("🔍 Top Keywords")
    text = ' '.join(filtered_df['description'].str.lower())

    # Check if there's any text data before generating the word cloud
    if text.strip():  # If text is not empty or only whitespace
        stop_words = set(stopwords.words('english'))
        wordcloud = WordCloud(width=800, height=400,
                              background_color='white',
                              stopwords=stop_words,
                              colormap='viridis').generate(text)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        st.pyplot(plt, use_container_width=True)
    else:
        st.write("No text data available for the word cloud.")

    # Dish Popularity
    st.subheader("🍽️ Dish Mentions")
    dish_counts = filtered_df[dish_keywords].sum().sort_values(ascending=False)
    fig = px.bar(dish_counts,
                 orientation='h',
                 labels={'value': 'Mentions', 'index': 'Dish'},
                 color=dish_counts.values,
                 color_continuous_scale='Blues')
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

# ======================
# Advanced Insights
# ======================
st.header("🚀 Predictive Insights")

# Prediction Explanation
st.subheader("🔮 Low Rating Risk Factors")
coefs = pd.DataFrame({
    'word': vectorizer.get_feature_names_out(),
    'coef': model.coef_[0]
}).sort_values('coef', ascending=False)

top_risk_factors = coefs.head(5).set_index('word')
top_positive_factors = coefs.tail(5).set_index('word')

risk_col1, risk_col2 = st.columns(2)
risk_col1.markdown("**Top Risk Factors**")
risk_col1.dataframe(top_risk_factors.style.background_gradient(cmap='Reds'), use_container_width=True)

risk_col2.markdown("**Top Positive Factors**")
risk_col2.dataframe(top_positive_factors.style.background_gradient(cmap='Greens'), use_container_width=True)

# ======================
# Review Summaries
# ======================
st.header("📋 Review Summaries")

# Display review summaries in a new column
st.dataframe(filtered_df[['customer_name', 'ratings', 'dates', 'review_summary']].sort_values('dates', ascending=False),
             hide_index=True,
             column_config={
                 "dates": st.column_config.DateColumn("Date"),
                 "ratings": st.column_config.NumberColumn("Rating", format="%d ⭐"),
                 "review_summary": st.column_config.TextColumn("Review Summary")
             },
             use_container_width=True)

# ======================
# Raw Data Explorer
# ======================
st.header("📋 Data Explorer")
st.dataframe(filtered_df[['customer_name', 'ratings', 'dates', 'description']].sort_values('dates', ascending=False),
             hide_index=True,
             column_config={
                 "dates": st.column_config.DateColumn("Date"),
                 "ratings": st.column_config.NumberColumn("Rating", format="%d ⭐")
             },
             use_container_width=True)

# ======================
# Footer
# ======================
st.markdown("---")

# Display the AI-generated review summary
st.subheader("🤖 AI-Generated Summary Review")
summary = generate_summary_review(filtered_df)
st.write(summary)

# Run with: streamlit run tiffinwala_dashboard.py
