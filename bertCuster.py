from sklearn.datasets import fetch_20newsgroups
import pandas as pd
import numpy as np
# Creating a dataframe from the data imported
dataset = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))['data']
print(len(dataset)) # the length of the data
print(type(dataset)) # the type of variable the data is stored in
print(dataset[:2]) # the first instance of the content within the data
full_train = pd.DataFrame()
full_train['text'] = dataset
full_train['text'] = full_train['text'].fillna('').astype(str) # removing any nan type objects
full_train
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# If the following packages are not already downloaded, the following lines are needed
# nltk.download('wordnet')
# nltk.download('omw-1.4')
# nltk.download('punkt')

filtered_text = []

lemmatizer = WordNetLemmatizer()

for i in range(len(full_train)):
    text = lemmatizer.lemmatize(full_train.loc[i, 'text'])
    text = text.replace('\n', ' ')
    filtered_text.append(text)

filtered_text[:1]
i=0
from bertopic import BERTopic
import torch
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
from bertopic.vectorizers import ClassTfidfTransformer
# Step 1 - Extract embeddings
embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
# Step 2 - Reduce dimensionality
umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine')
# Step 3 - Cluster reduced embeddings
hdbscan_model = HDBSCAN(min_cluster_size=15, metric='euclidean', cluster_selection_method='eom', prediction_data=True)
from sklearn.feature_extraction.text import CountVectorizer
# Step 4 - Tokenize topics
vectorizer_model = CountVectorizer(stop_words="english")
# Step 5 - Create topic representation
ctfidf_model = ClassTfidfTransformer()
topic_model = BERTopic(
    embedding_model=embedding_model,    # Step 1 - Extract embeddings
    umap_model=umap_model,              # Step 2 - Reduce dimensionality
    hdbscan_model=hdbscan_model,        # Step 3 - Cluster reduced embeddings
    vectorizer_model=vectorizer_model,  # Step 4 - Tokenize topics
    ctfidf_model=ctfidf_model,          # Step 5 - Extract topic words
    diversity=0.5,                      # Step 6 - Diversify topic words
    nr_topics=10
)
topics, probabilities = topic_model.fit_transform(filtered_text)
topic_model.get_document_info(filtered_text)
topic_model.get_topic_freq()
topic_model.get_topic(0)
topic_model.visualize_barchart()
embeddings = embedding_model.encode(filtered_text, show_progress_bar=False)

# Run the visualization with the original embeddings
topic_model.visualize_documents(filtered_text, embeddings=embeddings)
topic_model.visualize_hierarchy()

