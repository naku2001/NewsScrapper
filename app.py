import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
#import matplotlib.pyplot as plt

combined_df = pd.read_csv("data.csv")


# Vectorize 'category'
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(combined_df['category'])

# KMeans clustering
kmeans = KMeans(n_clusters=4)
kmeans.fit(X)
combined_df['cluster'] = kmeans.labels_

# Cluster names
cluster_names = {0: "Business", 1: "Politics", 2: "Arts", 3: "Sport"}
combined_df['cluster_name'] = combined_df['cluster'].map(cluster_names)

# Streamlit App
st.title("Clustering Platform")
st.markdown("This web displays categories of related stories.")



# Cluster selection
selected_cluster = st.selectbox("Select a cluster", combined_df['cluster_name'].unique())
cluster_data = combined_df[combined_df['cluster_name'] == selected_cluster]

# Display related stories
st.subheader("Related Stories:")
for url in cluster_data['link']:
    st.write(url)

# Cluster analysis
st.subheader("Cluster Analysis:")
for cluster_name in combined_df['cluster_name'].unique():
    cluster_data = combined_df[combined_df['cluster_name'] == cluster_name]
    st.write(f"Cluster {cluster_name}:")
    st.write(f"- Number of articles: {len(cluster_data)}")

    # Handle missing columns gracefully
    if 'section' in combined_df.columns:
        st.write(f"- Top categories: {cluster_data['section'].value_counts().head(3)}")
    if 'title' in combined_df.columns:
        st.write(f"- Top keywords: {cluster_data['title'].str.split(',').explode().value_counts().head(3)}")


