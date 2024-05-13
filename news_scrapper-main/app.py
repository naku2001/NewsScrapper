import streamlit as st
import requests
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from nltk.tokenize import word_tokenize

nltk.download('punkt')

def fetch_news_api():
  """Fetches news articles from the News API."""
  url = "https://newsapi.org/v2/top-headlines"
  params = {
      "country": "us",
      "apiKey": "YOUR_NEWS_API_KEY"  # Replace with your News API key
  }
  response = requests.get(url, params=params)
  data = response.json()
  articles = []
  for article in data.get('articles', []):
    title = article.get('title')
    description = article.get('description')
    url = article.get('url')
    if title and description and url:
      articles.append({
          "title": title,
          "description": description,
          "url": url
      })
  return articles

def preprocess_text(articles):
  """Preprocesses text data by tokenizing and applying TF-IDF."""
  if not articles:
    return None, None
  vectorizer = TfidfVectorizer(stop_words='english', tokenizer=word_tokenize)
  article_texts = [article["title"] + " " + article["description"] for article in articles]
  X = vectorizer.fit_transform(article_texts)
  return X, article_texts

def k_means_cluster(X, num_clusters=5):
  """Performs K-means clustering on the TF-IDF vectors."""
  kmeans = KMeans(n_clusters=num_clusters, random_state=42)
  kmeans.fit(X)
  return kmeans

def get_cluster_labels(kmeans_model, articles):
  """Assigns cluster labels to articles based on K-means results."""
  labels = kmeans_model.labels_
  clusters = {}
  for i, label in enumerate(labels):
    if label not in clusters:
      clusters[label] = []
    clusters[label].append(articles[i])
  return clusters

def categorize_clusters(clusters):
  """Categorizes clusters based on keyword presence in articles."""
  categories = {}
  for cluster, articles in clusters.items():
    category = get_category(articles)
    if category not in categories:
      categories[category] = {}
    categories[category][cluster] = articles
  return categories

def get_category(articles):
  """Assigns a category (e.g., Politics, Sports) to a cluster."""
  keywords = {
      "Politics": ["politic", "president", "congress", "government", "election", "vote", "senate", "democrat", "republican", "parliament"],
      "Sports": ["sports", "football", "soccer", "basketball", "baseball", "hockey", "tennis", "olympics", "athlete", "game"],
      "Business": ["business", "economy", "market", "stock", "finance", "company", "industry", "trade", "economic", "market"],
      "Technology": ["technology", "tech", "innovation", "digital", "internet", "gadget", "device", "software", "hardware", "computer"],
      "Health": ["health", "medical", "doctor", "hospital", "disease", "vaccine", "medicine", "patient", "wellness", "care"]
  }
  scores = {category: 0 for category in keywords}
  for article in articles:
    title = article['title'].lower()
    description = article['description'].lower()
    for category, words in keywords.items():
      for word in words:
        if word in title or word in description:
          scores[category] += 1
  return max(scores, key=scores.get)

def main():
  """Main function for the Streamlit application."""
  st.title("News Article Clustering")

  # Fetch news articles
  articles = fetch_news_api()
  if not articles:
    st.error("Failed to fetch news articles. Please check your API key or try again later.")
    return

  # Preprocess text data
  X, article_texts = preprocess_text(articles)
  if X is None or article_texts is None:
    st.error("Failed to")
