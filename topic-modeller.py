import numpy as np
import pandas as pd
import os
# Language detection
import cld3
# Data visualisation
import matplotlib.pyplot as plt
import seaborn as sns
# Preprocessing
import re
from autocorrect import Speller
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
# Embedding
from sentence_transformers import SentenceTransformer
# Dimensionality reduction
from umap import UMAP
# Clustering
from hdbscan import HDBSCAN
# Tokenisation
from sklearn.feature_extraction.text import CountVectorizer
# Topic modelling
from bertopic import BERTopic
# Weighting
from bertopic.vectorizers import ClassTfidfTransformer
# Topic representation
from bertopic.representation import MaximalMarginalRelevance
# Coherence scoring
from gensim.models.coherencemodel import CoherenceModel
from gensim import corpora
# Serialisation
import joblib

# Download list of stopwords
nltk.download('stopwords')

# Set Matplotlib defaults
sns.set_style("darkgrid")
plt.rc("figure", autolayout=True)
plt.rc(
    "axes",
    labelweight="bold",
    labelsize="large",
    titleweight="bold",
    titlesize=14,
    titlepad=10,
)

# We use coherence score to evaluate performance of the topic models
def calculate_coherence_score(topic_model, docs):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    cleaned_docs = topic_model._preprocess_text(docs)
    vectorizer = topic_model.vectorizer_model
    analyzer = vectorizer.build_analyzer()
    tokens = [analyzer(doc) for doc in cleaned_docs]
    dictionary = corpora.Dictionary(tokens)
    corpus = [dictionary.doc2bow(token) for token in tokens]
    topics = topic_model.get_topics()
    topic_words = [[word for word, _ in topic_model.get_topic(topic)] for topic in topics]
    topic_words = [words for words in topic_words if len(words) > 0]

    coherence_model = CoherenceModel(topics=topic_words,
                                    texts=tokens,
                                    corpus=corpus,
                                    dictionary=dictionary,
                                    coherence='c_v')
    coherence = coherence_model.get_coherence()

    return coherence

# We define a training function for our topic model for automated hyperparameter tuning
def train_bert(param_vals):
    topic_model = BERTopic(embedding_model=embedding_model,
                           umap_model=UMAP(n_neighbors=param_vals['n_neighbors'], n_components=param_vals['n_components'], min_dist=param_vals['min_dist'], metric='cosine', low_memory=False, random_state=23),
                           hdbscan_model=HDBSCAN(min_cluster_size=param_vals['min_cluster_size'], min_samples=int(param_vals['min_samples_mult'] * param_vals['min_cluster_size']), metric=param_vals['metric'], cluster_selection_method='eom', prediction_data=True),
                           vectorizer_model=CountVectorizer(stop_words=stopwords.words('english'), ngram_range=param_vals['ngram_range'], min_df=param_vals['min_df'], max_features=10_000),
                           ctfidf_model=ClassTfidfTransformer(bm25_weighting=True, reduce_frequent_words=True),
                           representation_model=MaximalMarginalRelevance(diversity=param_vals['diversity']),
                           top_n_words=param_vals['top_n_words'],
                           min_topic_size=param_vals['min_topic_size'],
                           language='english')

    topics, probs = topic_model.fit_transform(docs, embeddings)
    
    return topic_model

# Load data
amazon = pd.read_csv('./input/amazon-product-reviews-dataset/7817_1.csv')

# Select columns relevant to identifying topics in the reviews
reviews = amazon.loc[:, ['reviews.title', 'reviews.text']]
reviews = reviews.rename(columns={'reviews.title': 'title', 'reviews.text': 'text'})

# Combine title and body of reviews into single documents
reviews.loc[:,'title'] = reviews.title.fillna('')
reviews['document'] = reviews.title + ' ' + reviews.text
reviews = reviews.drop(['title', 'text'], axis=1)

# Remove non-alphabetical characters
for i in range(len(reviews)):
    reviews['document'].iloc[i] = re.sub(r"[^a-zA-Z]+", ' ', reviews['document'].iloc[i]).lower().strip()

# Remove duplicate reviews and convert to a list of strings
docs = reviews.document.unique().tolist()

# Remove non-English reviews
docs = [doc for doc in docs if cld3.get_language(doc).language == 'en']

# Correct typos/misspellings in reviews
spell = Speller()
for i in range(len(docs)):
    docs[i] = ' '.join([spell(w) for w in word_tokenize(docs[i])])

# Pre-compute embeddings
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = embedding_model.encode(docs)

# Define the hyperparameter ranges for tuning the topic model
param_ranges = {'min_dist': [0.0, 0.1, 0.25, 0.5], 'n_neighbors': [15,25,50], 'n_components': [3,5,10,15,20,30], 
                'min_cluster_size': [5,10,15,20], 'min_samples_mult': [0.6,0.8,1.0,1.2,1.4], 'metric': ['euclidean'],
                'ngram_range': [(1,1), (1,2), (1,3)], 'min_df': [1,2,3,5,10],
                'diversity': [0.1,0.3,0.5,0.7,0.9],
                'top_n_words': [10,15,20], 'min_topic_size': [5,10,15,20]}

top_5 = [(BERTopic(), 0, {})] * 5

# Cycle through 1000 random combinations of hyperparameter values chosen from the ranges above, choosing the models with the top 5 coherence scores.
# WARNING: This will take a long time! Try replacing 1000 with 500 or 200 or even 100 if it is taking too long.
for i in range(1000):
    param_vals = {param: param_ranges[param][np.random.randint(len(param_ranges[param]))] for param in param_ranges}
    try:
        topic_model = train_bert(param_vals)
        score = calculate_coherence_score(topic_model, docs)
    except Exception:
        continue
    if score > top_5[-1][1] and len(topic_model.get_topics()) > 3:
        top_5[-1] = (topic_model, score, param_vals)
        top_5 = sorted(top_5, key=lambda x: x[1], reverse=True)
        
print(f'Top 5 scores: {top_5[0][1]}, {top_5[1][1]}, {top_5[2][1]}, {top_5[3][1]}, {top_5[4][1]}')
print(f'Top 5 parameter values:\n {top_5[0][2]},\n {top_5[1][2]},\n {top_5[2][2]},\n {top_5[3][2]},\n {top_5[4][2]}')

# Save the cleaned data and the top 5 topic models.
# You can then perform data visualisation on these to identify the one that looks best.
# For example, use `topic_model.visualize_barchart()` and `topic_model.visualize_documents(docs)`.
joblib.dump(docs, 'clean_docs')
for i in range(5):
    joblib.dump(top_5[i][0], f'topic_model_{i}')