# -*- coding: utf-8 -*-
import string
import os
import sys
import nltk
import gensim
import threading
import platform
import wordcloud
import spacy
from bs4 import BeautifulSoup  # Web page parsing and data acquisition
import re  # Regular expressions for text matching
import urllib.request, urllib.error  # Make URL and get web page data
import en_ner_bc5cdr_md
import pandas as pd
import xml.etree.ElementTree as ET
from nltk.tokenize import word_tokenize
from gensim.corpora.dictionary import Dictionary
from nltk.stem import WordNetLemmatizer, SnowballStemmer
import matplotlib.pyplot as plt
import time
from 废弃asynicoFTp import getFTP
from getText import getArtcle
import numpy as np
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from scipy.sparse import csr_matrix
docs={}
t1 = time.time()

asd=['PMC7323196', 'PMC9043892', 'PMC8504968', 'PMC7915126', 'PMC8238037', 'PMC9380879', 'PMC8524328', 'PMC7405836']
##get xml use asynico
# getFTP(asd)

t2 = time.time()
print(int(t2-t1))
##get text form xml use bioc
docs=getArtcle()
stpwrd = nltk.corpus.stopwords.words('english')
# joining texts of each article into one string.
docs_list = [docs.get(doc) for doc in docs]

# removing whitespace
data = [re.sub(r'\s', ' ', doc) for doc in docs_list]

# removing urls:
# https:\/\/www\.\w+\.\w+
data = [re.sub(r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b', ' ', doc) for doc in data]
# removing numbers
# r'[\s\(][^-a-zA-Z]+\-*[\d\.]+'
data = [re.sub(r'[\s\(][^-a-zA-Z]+\-*[^-a-zA-Z]+', ' ', doc) for doc in data]

# Adding 2019 to -nCoV:
data = [re.sub(r'-nCoV', '2019-nCoV', doc) for doc in data]
data = [re.sub(r'-CoV', '2019-CoV', doc) for doc in data]
# Removing medical units:
data = [re.sub(r'[a-zA-Z]+\/[a-zA-Z]+', '', doc) for doc in data]

# Removing white spaces again
data = [re.sub(r'\s', ' ', doc) for doc in data]

# removing punctuations:
# removing '-' from punctuations list.
puncs = re.sub('-', '', string.punctuation)
data = [re.sub(r'[{}]+'.format(puncs), '', doc) for doc in data]
pattern = r'[A-Z]{1}[a-z]{2,}\s'  # Defined pattern for finding capital words except those which contain digits

for i, doc in enumerate(data):
    index_temp = [(m.start(0), m.end(0)) for m in re.finditer(pattern, doc)]
    for ind in index_temp:
        ii = ind[0]
        jj = ind[1]

        data[i] = data[i].replace(data[i][ii:jj], data[i][ii:jj].lower())
# =============================================================================
stemmer = SnowballStemmer("english")
wnl = WordNetLemmatizer()

# A function for lemmatizing and stemming a text
def lemmatize_stemming(text):
    return stemmer.stem(wnl.lemmatize(text, pos='v'))


# A token preprocessing function
def preprocess(text):
    result = []
    mydict = {}  # A dictionary which will contain original tokens before lemmatizing and stemming
    for token in word_tokenize(text):
        # if token not in stpwrd and len(token) >= 3:
        if len(token) >= 2:
            temp = lemmatize_stemming(token)
            mydict[temp] = token
            result.append(temp)
    return result, mydict


mywords = []
# A dictionary which contains original tokens as value and lemmetized stemmized token as key:
token_word_dict = {}

for doc in data:
    data_new = []

    data_new=((doc).split(" "))
    tagged = nltk.pos_tag(data_new)
    data_new1 = []
    for word, pos in tagged:
        if pos != 'MD':
            data_new1.append(word)
    var = ' '.join(data_new1)
    mywords.append(preprocess(var)[0])
    token_word_dict.update(preprocess(var)[1])
            # print(preprocess(doc)[1])
# Removing words with frequency < 2:
# for sub in mywords:
#     sub[:] = [ele for ele in sub if sub.count(ele) > 1]
# token_word_dict = [x for x in token_word_dict if x != None]
# Building the bigram models



bigram = gensim.models.phrases.Phrases(mywords, min_count=2, threshold=10)

# cearting list of bigrams:
mywords2 = bigram[mywords]

# Building the trigram models
trigram = gensim.models.phrases.Phrases(bigram[mywords], min_count=2, threshold=10)
mywords3 = trigram[mywords2]



# A function for removing stop words:
def remove_stopwrd(txt):
    result = []
    for wrd in txt:
        temp = wrd.split('_')
        if not any(ele in stpwrd for ele in temp):
            result.append(wrd)
    return result


mywords3_no_stopwrd = [[] for i in range(len(mywords3))]

mywords3_no_stopwrd = [remove_stopwrd(lis) for lis in mywords3]

# Create Dictionary of trigrams

dictionary_trigram = Dictionary(mywords3_no_stopwrd)


# Create Corpus
corpus_trigram= [dictionary_trigram.doc2bow(text) for text in mywords3_no_stopwrd]
# =============================================================================
tfidf_transformer = TfidfTransformer()
rows = []
cols = []
vals = []

for i, row in enumerate(corpus_trigram):
    for j, val in row:
        rows.append(i)
        cols.append(j)
        vals.append(val)

# 转换为CSR稀疏矩阵类型
csr_mat = csr_matrix((vals, (rows, cols)))
X_tfidf = tfidf_transformer.fit_transform(csr_mat)
n_topics = 10
lda = LatentDirichletAllocation(n_components=n_topics, max_iter=50,
                                learning_method='batch',
                                learning_offset=50,
                                #                                 doc_topic_prior=0.1,
                                #                                 topic_word_prior=0.01,
                                random_state=0)
lda.fit(X_tfidf)


def print_top_words(model, feature_names, n_top_words):
    tword = []
    tword2 = []
    tword3=[]
    for topic_idx, topic in enumerate(model.components_):
        print("Topic #%d:" % topic_idx)
        topic_w = [feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]
        topic_pro=[str(round(topic[i],3)) for i in topic.argsort()[:-n_top_words - 1:-1]]  #(round(topic[i],3))
        tword.append(topic_w)
        tword2.append(topic_pro)
        print(" ".join(topic_w))
        print(" ".join(topic_pro))
        print(' ')
        word_pro=dict(zip(topic_w,topic_pro))
        tword3.append(word_pro)
    return tword3

##输出每个主题对应词语和概率
n_top_words = 8
array_data = np.array(list(dictionary_trigram.token2id.keys()))

word_pro = print_top_words(lda, array_data, n_top_words)



t2 = time.time()
print(int(t2-t1))
i=0