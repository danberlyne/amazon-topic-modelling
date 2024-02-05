# Amazon topic modelling

This is a data science project based on the [Amazon Product Reviews Dataset](https://www.kaggle.com/datasets/yasserh/amazon-product-reviews-dataset). Our goal is to identify the major topics in the customer reviews of various products.

## The data

The `input` folder contains the following data.

1. `7817_1.csv`, which contains the Amazon product reviews data.
2. `amazon_topic_model`, which contains the optimised version of the BERTopic model that we built during this project. This was saved as a file via the `joblib` package and can be loaded with `joblib.load()`. Visualisation and analysis of this model can be found in the Conclusion section of the notebook `amazon-topic-modelling.ipynb`. **Note:** You need to download the notebook and view it locally in order to see the Plotly visualisations.
3. `clean_docs`, which contains the cleaned version of the Amazon product reviews data, as a list of strings. This was saved as a file using `joblib`.

## Methodology

Below is a brief summary of the methodology. A more detailed methodology, complete with code snippets and plots, can be found in the notebook `amazon-topic-modelling.ipynb`.

Our goal is to perform topic modelling on the Amazon product reviews using `BERTopic`, identifying an optimal number of topics to split the data into, and identifying what the major topics in the reviews are. We apply the following techniques to achieve this.

1. **Preprocessing.** We first identify the title and body of each review in the dataset and compile them into single documents. We then remove non-alphabetical characters, remove duplicate reviews, identify the languages of the reviews and remove any reviews that are not in English. Finally, we correct any typos/misspellings in the reviews.
2. **Topic modelling.** We use `BERTopic` for our topic modelling. This involves multiple stages, each with its own separate model: embedding via `SentenceTransformer` using the `all-MiniLM-L6-v2` LLM; dimensionality reduction via `UMAP`, clustering via `HDBSCAN`, tokenisation via `CountVectorizer`, weighting via `ClassTfidfTransformer`, and topic representation via `MaximalMarginalRelevance`. We start by using the default settings for each model to establish a baseline topic model.
3. **Hyperparameter tuning.** We tune the hyperparameters of the models listed above by iterating through 1000 random combinations of values, feeding them into `BERTopic`, and calculating the coherence score of the resulting topics. We pick the five highest coherence scores and display visualisations for the corresponding topic models, so that we can manually pick the best one.

## Results

Running the script `topic-modeller.py` consistently achieves coherence scores between 0.71 and 0.76. 

We identify six major topics: *Voice Assistants*, *Television*, *Reading*, *Tablets*, *Accessories*, and *Audio*. These are separated into distinct clusters in the dataset, with a potential *Remote Controls* sub-cluster of the *Television* topic. 

See the `amazon_topic_model` file in the `inputs` folder for a sample optimised model constructed using this code. Load the `amazon_topic_model` and `clean_docs` files with `joblib.load()` and use the `visualize_barchart()` and `visualize_documents()` methods in `BERTopic` to visualise the topics. You can also see these visualisations in the Conclusion section of the notebook `amazon-topic-modelling.ipynb` (you need to download the notebook and view it locally in order to see the Plotly visualisations). 
