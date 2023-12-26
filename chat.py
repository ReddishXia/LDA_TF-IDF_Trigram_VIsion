from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Example preprocessed documents
documents = [
    "This is the first document",
    "This document is the second document",
    "And this is the third one",
    "Is this the first document"
]

# Create the tf-idf matrix
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(documents)

# Fit LDA model
num_topics = 2  # Define the number of topics
lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
lda.fit(tfidf_matrix)

# Extract topics and word distributions
feature_names = vectorizer.get_feature_names_out()
for topic_idx, topic in enumerate(lda.components_):
    print("Topic %d:" % (topic_idx))
    words = [feature_names[i] for i in topic.argsort()[:-10 - 1:-1]]  # Top 10 words per topic
    print(", ".join(words))

# Assign topics to documents
doc_topic_probabilities = lda.transform(tfidf_matrix)
for doc_idx, doc in enumerate(doc_topic_probabilities):
    print("Document %d: Topic probabilities" % (doc_idx))
    for topic_idx, prob in enumerate(doc):
        print("Topic %d: %f" % (topic_idx, prob))
