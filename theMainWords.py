# -*- coding: utf-8 -*-

# Created on Fri Jun 18 09:19:53 2021


import os
import nltk
import gensim
import re
import string
import wordcloud
import xml.etree.ElementTree as ET
from nltk.tokenize import word_tokenize
from gensim.corpora.dictionary import Dictionary
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from scipy.sparse import csr_matrix
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.decomposition import LatentDirichletAllocation
# nltk.download('averaged_perceptron_tagger')
rootPath = r"C:\Users\79988\PycharmProjects\chatGPT_LDA_Trigram\test"
resultPath = r"C:\Users\Administrator\Desktop"
xml_papers = os.listdir(rootPath)
judgeShort=1
start = time.perf_counter()
def delete_repeat_max(txt,repeat_wrd,repeat_len):
    txt = [x for x in txt if x != None]
    for wrd in txt:
        temp_length = 0
        for wrd1 in txt:
            if wrd1.find(wrd)!=-1:
                length=len(wrd1)
                if length>temp_length:
                    temp_length=length
        if temp_length!=len(wrd):
            repeat_wrd.append(wrd)
            repeat_len.append(temp_length)


def delete_repeat(txt,repeat_wrd,repeat_len):
    for i,wrd in enumerate(txt):
        if len(repeat_wrd)!=0:
            for k,wrd2 in enumerate(repeat_wrd):
                if wrd.find(wrd2)!=-1:
                    length=len(wrd)
                    if length<repeat_len[k]:
                        if txt.count(wrd)!=0:
                            txt[i]=""


    while "" in txt:
        txt.remove("")
def begin_Time():
    global start
    start = time.perf_counter()
def used_Time():
    global start
    end = time.perf_counter()
    runTime = end - start
    print("运行时间：", runTime, "秒")
    start = time.perf_counter()
def find_full_name(text, abbreviation):
    # 通过正则表达式匹配MINJ所在的短语
    length=len(abbreviation)
    match = re.search(r'\b(\w+\W+){0,%d}%s\b' % (length, abbreviation), text)

    if match:
        # 获取匹配到的短语，并将其中的括号和MINJ去掉

        phrase = match.group(0).replace('('+abbreviation, '').strip()
        # 获取短语中的前四个单词
        words = phrase.split()[:length]
        words.reverse()  # 翻转列表
        index =  words.index(next(word for word in words if word.startswith(abbreviation[0]))) +1
        # 找到第一个首字母为'M'的单词在列表中的位置
        result = words[:index]  # 输出在这个单词后的所有单词
        result.reverse()  # 翻转列表
        s = ''.join(result)
        s_set = [char for char in s.lower()]

        target_set = [char for char in abbreviation.lower()]

        index=0
        intersect = []
        for letter in target_set:
            for letter1 in range(index, len(s_set)):
                index+=1
                if letter in s_set:
                    intersect.append(letter)
                    break
        # 判断交集的长度是否大于等于 4
        if len(intersect) == len(abbreviation):
            return True
        else:
            return False
##除去重复的单词
def delete_repeat_max(txt,repeat_wrd,repeat_len):
    txt = [x for x in txt if x != None]
    for wrd in txt:
        temp_length = 0
        for wrd1 in txt:
            if wrd1.find(wrd)!=-1:
                length=len(wrd1)
                if length>temp_length:
                    temp_length=length
        if temp_length!=len(wrd):
            repeat_wrd.append(wrd)
            repeat_len.append(temp_length)


def delete_repeat(txt,repeat_wrd,repeat_len):
    for i,wrd in enumerate(txt):
        if len(repeat_wrd)!=0:
            for k,wrd2 in enumerate(repeat_wrd):
                if wrd.find(wrd2)!=-1:
                    length=len(wrd)
                    if length<repeat_len[k]:
                        if txt.count(wrd)!=0:
                            txt[i]=""


    while "" in txt:
        txt.remove("")

if "desktop.ini" in xml_papers:
    xml_papers.remove("desktop.ini")  # removing the hidden 'desktop.ini' which will cause issue

# A dictionary that will contain the PMC IDs as keys and texts of articles sections as value:
# docs = dict.fromkeys(xml_papers)

# will contain articles after parsing
articles = [[] for i in range(len(xml_papers))]

# A dictionary that will contain section types of the articles

Important_sections = ['ABSTRACT', 'INTRO', 'METHODS', 'DISCUSS', 'RESULTS', 'CASE', 'CONCL', 'ABBR', 'FIG', 'TABLE']
Other_sections = ['SUPPL', 'REF', 'APPENDIX', 'AUTH_CONT', 'ACK_FUND', 'COMP_INT', 'REVIEW_INFO']

stpwrd = nltk.corpus.stopwords.words('english')

# Adding new stop words to the list of stop words:
new_stopwords = ["surname", "ref", "abstract", "intro", "http", 'left upper', 'right upper', 'article',
                 'published', 'even though', 'paragraph', 'page', 'sentence', 'et', 'al', 'etc','province','would','today',]
stpwrd.extend(new_stopwords)

# Parsing the XML files and getting its root
xml_papersw=[]
##############################################################
##############################################################
for k, article in enumerate(xml_papers):
    modified_path = os.path.join(rootPath, article)
    temp = ET.parse(modified_path, ET.XMLParser(encoding='utf-8'))
    articles[k].append(temp)
    # print(temp)
    collection = temp.getroot()
    for i, document in enumerate(collection):
        judgeShort = 0
        for x in document.findall("passage"):
            for inf in x.findall('infon'):
                if inf.attrib == {'key': 'section_type'}:
                    if inf.text not in Other_sections:
                        if inf.text in ['ABSTRACT', 'CONCL','METHODS','RESULTS']:
                            judgeShort=1
    list_i = list(xml_papers[k])  # str -> list
    list_i.insert(10, str(judgeShort))  # 注意不用重新赋值
    xml_papersw1= ''.join(list_i)
    xml_papersw.append(xml_papersw1)
#############################################################
############################################################
docs = dict.fromkeys(xml_papersw)
section_types = dict.fromkeys(xml_papersw)

for k, article in enumerate(xml_papers):
    modified_path = os.path.join(rootPath, article)
    temp= ET.parse(modified_path, ET.XMLParser(encoding='utf-8'))
    articles[k].append(temp)
    # print(temp)
    collection = temp.getroot()
    section_types[xml_papersw[k]] = []
    docs[xml_papersw[k]] = []
    # Extracting all the texts of all sections
    for i, document in enumerate(collection):
        judgeShort = 0
        for x in document.findall("passage"):
            # print(x.findall('infon'))
            infon_list = x.findall('infon')

            # Removing footnote and table contents sections:
            if any(inf.text == 'footnote' for inf in infon_list) or any(inf.text == 'table' for inf in infon_list):
                document.remove(x)
    for x in document.findall("passage"):
            for inf in x.findall('infon'):
                if inf.attrib == {'key': 'section_type'}:
                    section_types[xml_papersw[k]].append(inf.text)
                    if inf.text not in Other_sections:
                        temp1 = getattr(x.find('text'), 'text', None)
                        if inf.text in ['ABSTRACT', 'CONCL']:
                            docs[xml_papersw[k]].append(temp1)
                        else:
                            docs[xml_papersw[k]].append(temp1)

    docs[xml_papersw[k]] = list(filter(None, docs[xml_papersw[k]]))

# list(docs.keys()).index('PMC7084952.xml')

# joining texts of each article into one string.
docs_list = [' '.join(docs.get(doc)) for doc in docs]

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

# lowering new line capital words except those which contain digits:
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

begin_Time()

def return_myworld():
    return mywords


bigram = gensim.models.phrases.Phrases(mywords, min_count=2, threshold=10)
# cearting list of bigrams:
mywords2 = bigram[mywords]

# Building the trigram models
trigram = gensim.models.phrases.Phrases(bigram[mywords], min_count=2, threshold=10)
mywords3 = trigram[mywords2]

fourgram = gensim.models.phrases.Phrases(trigram[mywords2], min_count=2, threshold=10)
mywords4 = fourgram[mywords3]

tiftgram = gensim.models.phrases.Phrases(fourgram[mywords3], min_count=2, threshold=10)
mywords5 = trigram[mywords4]

used_Time()

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

mywords5_no_stopwrd = [[] for i in range(len(mywords5))]

mywords5_no_stopwrd = [remove_stopwrd(lis) for lis in mywords5]

# Create Dictionary of trigrams

dictionary_trigram = gensim.corpora.Dictionary(mywords3_no_stopwrd)

dictionary_trigram5 = gensim.corpora.Dictionary(mywords5_no_stopwrd)
used_Time()

corpus_trigram= [dictionary_trigram.doc2bow(text) for text in mywords3_no_stopwrd]
tfidf_trigram_model = gensim.models.tfidfmodel.TfidfModel(corpus=corpus_trigram,
                                                          id2word=dictionary_trigram,
                                                          normalize=True)
temp3=[]
for i, corp in enumerate(corpus_trigram):
    temp3.append(tfidf_trigram_model[corp])

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
n_topics1 = 1
lda = LatentDirichletAllocation(n_components=n_topics1, max_iter=50,
                                learning_method='batch',
                                learning_offset=50,
                                #                                 doc_topic_prior=0.1,
                                #                                 topic_word_prior=0.01,
                                random_state=0)
lda.fit(X_tfidf)
used_Time()

def perplexity(tf):
    plexs = []
    scores = []
    n_max_topics = 10
    for i in range(1, n_max_topics):
        print('正在进行第', i, '轮计算')
        lda1 = LatentDirichletAllocation(n_components=i, max_iter=50,
                                        learning_method='batch',
                                        learning_offset=50, random_state=0)
        lda1.fit(tf)
        plexs.append(lda1.perplexity(tf))
        scores.append(lda1.score(tf))
    n_t = 9  # 区间最右侧的值。注意：不能大于n_max_topics
    x = list(range(1, n_t))
    plt.plot(x, plexs[1:n_t])
    plt.xlabel("number of topics")
    plt.ylabel("perplexity")
    plt.show()

perplexity(X_tfidf)



def print_top_words(model, feature_names, n_top_words):
    tword = []
    tword2 = []
    tword3=[]
    for topic_idx, topic in enumerate(model.components_):
        print("Topic #%d:" % topic_idx)
        topic_w=[]
        for i in topic.argsort()[:-n_top_words - 1:-1]:
            temp5 = feature_names[i].split('_')
            temp6 = ''
            for ii, tex in enumerate(temp5):  # Rejoining the trigrams with '_' again
                temp6 = temp6 + token_word_dict.get(temp5[ii]) + ' '

            topic_w.append(temp6[:-1])
        # topic_w = [feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]
        topic_pro=[str(round(topic[i],3)) for i in topic.argsort()[:-n_top_words - 1:-1]]  #(round(topic[i],3))

        print(" ".join(topic_w))
        print(" ".join(topic_pro))
        print(' ')
        tword.append(topic_w)
        word_pro=dict(zip(topic_w,topic_pro))
        tword3.append(word_pro)
    return tword

word1=[]
word2=[]
word3=[]
word4=[]
word5=[]
def getOutNgram(p):
    for idx,data in enumerate(p):
        temp5 = data.split('_')
        temp6 = ''
        for ii, tex in enumerate(temp5):  # Rejoining the trigrams with '_' again
            temp6 = temp6 + token_word_dict.get(temp5[ii]) + ' '

        num = data.count('_')
        if num==0:
            word1.append(temp6)
        if num == 1:
            word2.append(temp6)
        if num == 2:
            word3.append(temp6)
        if num == 3:
            word4.append(temp6)
        if num == 4:
            word5.append(temp6)
##输出每个主题对应词语和概率
n_top_words = 100
array_data = np.array(list(dictionary_trigram.token2id.keys()))
array_data5 = np.array(list(dictionary_trigram5.token2id.keys()))
getOutNgram(array_data5.tolist())
# print(word1)
# print(word2)
# print(word3)
# print(word4)
# print(word5)
repeat_wrd=[[]]
repeat_len=[[]]
word_pro = print_top_words(lda, array_data, n_top_words)
delete_repeat_max(word_pro[0], repeat_wrd[0], repeat_len[0])
delete_repeat(word_pro[0], repeat_wrd[0], repeat_len[0])
word_proWithtagged = nltk.pos_tag(word_pro[0])
word_pro_deleteADJ = []
for word, pos in word_proWithtagged:
    if pos != 'MD'and pos !='VBG'and pos !='JJ'and pos !='IN'and pos !='VBD'and pos !='RB'and pos !='RBR'and pos !='RBS':
        word_pro_deleteADJ.append(word)
i=0
#
# # Create Corpus
# corpus_trigram= [dictionary_trigram.doc2bow(text) for text in mywords3_no_stopwrd]
# from gensim import corpora,models
# tfidf =models.TfidfModel(corpus_trigram,normalize=True)
# corpus_tfidf=tfidf[corpus_trigram]
#
# if __name__ == '__main__':
#     lda_model_tfidf = gensim.models.LdaMulticore(corpus_tfidf, num_topics=8, id2word=dictionary_trigram,
#                                                  chunksize=100, iterations=25,
#                                                  passes=2, workers=3
#                                                  )
#     print(lda_model_tfidf.print_topic(-1))