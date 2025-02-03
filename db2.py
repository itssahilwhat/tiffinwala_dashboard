# tiffinwala_dashboard.py
import streamlit as st
import pandas as pd
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from textblob import TextBlob
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import numpy as np
import json

# Download NLTK stopwords if not already downloaded
nltk.download('stopwords')

# Streamlit Setup
st.set_page_config(page_title="Tiffinwala Analytics", layout="wide", page_icon="🥘")
plt.style.use('ggplot')

# Load Data
@st.cache_data
def load_data():
    with open('tiffinwala_reviews.json') as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    df['dates'] = pd.to_datetime(df['dates'])
    return df

df = load_data()

# Dish Keywords (Ensure these are lowercase for consistent processing)
dish_keywords = [
    'dal', 'paneer', 'thepla', 'khandvi', 'undhiyu', 'kadhi', 'fafda', 'bajra rotla',
    'sev tameta nu shaak', 'ringan no olo', 'bhinda nu shaak', 'dudhi nu shaak',
    'batata nu shaak', 'lasaniya batata', 'karela nu shaak', 'dudhi chana nu shaak',
    'patra', 'khaman', 'handvo', 'puri', 'mooli paratha', 'masala khichdi', 'ghughra'
]

# Data Preprocessing
@st.cache_data
def enhance_data(df):
    # Sentiment analysis
    df['sentiment'] = df['description'].apply(lambda x: TextBlob(x).sentiment.polarity)
    df['sentiment_category'] = pd.cut(df['sentiment'],
                                      bins=[-1, -0.1, 0.1, 1],
                                      labels=['Negative', 'Neutral', 'Positive'])

    # Extract mentioned dishes (ensuring lowercase for comparison)
    df['description'] = df['description'].str.lower()

    for dish in dish_keywords:
        df[dish] = df['description'].str.contains(dish, case=False, na=False)

    return df

df = enhance_data(df)

# Train Simple Prediction Model
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
st.markdown("""
**Hackathon-ready analytics** for Tiffinwala's customer feedback data  
*1000+ reviews · Real-time insights · Predictive analytics*
""")

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

if len(filtered_df) > 0:
    metric_cols[0].metric("Total Reviews", len(filtered_df))
    metric_cols[1].metric("Avg Rating", f"{filtered_df['ratings'].mean():.1f} ⭐")
    positive_sentiment = f"{(filtered_df['sentiment_category'] == 'Positive').mean():.0%}"
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
    st.subheader("📈 Rating Trends")
    time_df = filtered_df.set_index('dates').resample('W')['ratings'].mean().reset_index()
    fig = px.line(time_df, x='dates', y='ratings', template='plotly_white',
                  labels={'ratings': 'Average Rating'}, height=300)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("😃😐😡 Sentiment Distribution")
    sentiment_counts = filtered_df['sentiment_category'].value_counts()
    fig = px.pie(sentiment_counts, names=sentiment_counts.index,
                 values=sentiment_counts.values, hole=0.4,
                 color_discrete_sequence=px.colors.qualitative.Pastel)
    st.plotly_chart(fig, use_container_width=True)

with viz_col2:
    st.subheader("🔍 Top Keywords")
    text = ' '.join(filtered_df['description'])

    if text.strip():
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

    st.subheader("🍽️ Dish Mentions")
    dish_columns = list(set(filtered_df.columns) & set(dish_keywords))
    if dish_columns:
        dish_counts = filtered_df[dish_columns].sum().sort_values(ascending=False)
        fig = px.bar(dish_counts, orientation='h',
                     labels={'value': 'Mentions', 'index': 'Dish'},
                     color=dish_counts.values, color_continuous_scale='Blues')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.write("No dish mentions found in the filtered data.")

# ======================
# Advanced Insights
# ======================
st.header("🚀 Predictive Insights")

st.subheader("🔮 Low Rating Risk Factors")
coefs = pd.DataFrame({'word': vectorizer.get_feature_names_out(),
                      'coef': model.coef_[0]}).sort_values('coef', ascending=False)

top_risk_factors = coefs.head(5).set_index('word')
top_positive_factors = coefs.tail(5).set_index('word')

risk_col1, risk_col2 = st.columns(2)
risk_col1.dataframe(top_risk_factors.style.background_gradient(cmap='Reds'))
risk_col2.dataframe(top_positive_factors.style.background_gradient(cmap='Greens'))

# ======================
# Footer
# ======================
st.markdown("---")
st.markdown("**Hackathon Ready Features:**  \n"
            "- Real-time predictive analytics  \n"
            "- Automated sentiment detection  \n"
            "- Interactive filters  \n"
            "- Risk factor identification")

