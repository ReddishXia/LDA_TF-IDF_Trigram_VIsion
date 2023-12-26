# -*- coding: utf-8 -*-

# Created on Fri Jun 18 09:19:53 2021
import math

import os
import json
import tomotopy as tp
import matplotlib
import nltk
import gensim
import re
import string
from gensim.models import LdaModel
import pyLDAvis.gensim
import pyLDAvis
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
# rootPath = r'E:\xmlcollection'
resultPath = r"C:\Users\Administrator\Desktop"
xml_papers = os.listdir(rootPath)
judgeShort=1
start = time.perf_counter()
xd=['PMC7824075.xml', 'PMC7824170.xml', 'PMC7824470.xml', 'PMC7824811.xml', 'PMC7824817.xml', 'PMC7825705.xml', 'PMC7825841.xml', 'PMC7825955.xml', 'PMC7826042.xml', 'PMC7827130.xml', 'PMC7827692.xml', 'PMC7827846.xml', 'PMC7827890.xml', 'PMC7827974.xml', 'PMC7828055.xml', 'PMC7828126.xml', 'PMC7828218.xml', 'PMC7828525.xml', 'PMC7828742.xml', 'PMC7829816.xml', 'PMC7829836.xml', 'PMC7829843.xml', 'PMC7829938.xml', 'PMC7830154.xml', 'PMC7830475.xml', 'PMC7830522.xml', 'PMC7830623.xml', 'PMC7830668.xml', 'PMC7830673.xml', 'PMC7831024.xml', 'PMC7831030.xml', 'PMC7831046.xml', 'PMC7831445.xml', 'PMC7831568.xml', 'PMC7831665.xml']
def tfidfProcess(mywords_no_stopwrd,article_Item,article_Name):

    dictionary_trigram = gensim.corpora.Dictionary(mywords_no_stopwrd)
    corpus_trigram = [dictionary_trigram.doc2bow(text) for text in mywords_no_stopwrd]
    tfidf_trigram_model = gensim.models.tfidfmodel.TfidfModel(corpus=corpus_trigram,
                                                              id2word=dictionary_trigram,
                                                              normalize=True)

    # Top 10 tokens
    # tfidf_top10_words=[[] for i in range(len(corpus_trigram))]
    repeat_wrd=[[] for i in range(len(corpus_trigram))]
    repeat_len = [[] for i in range(len(corpus_trigram))]
    top10_trigram_of_articles = [[] for i in range(len(corpus_trigram))]
    top_trigram_of_articles = [[] for i in range(len(corpus_trigram))]
    # Will contain the original words before being stemmized and lemmatized:
    top10_tri_words_original = [[] for i in range(len(corpus_trigram))]
    top10_tri_freqs = [[] for i in range(len(corpus_trigram))]
    top10_tri_words_original2 = [[] for i in range(len(corpus_trigram))]
    top10_tri_freqs2 = [[] for i in range(len(corpus_trigram))]
    top10_tri_words_original3 = [[] for i in range(len(corpus_trigram))]
    top10_tri_freqs3 = [[] for i in range(len(corpus_trigram))]
    top10_tri_words_original4 = [[] for i in range(len(corpus_trigram))]
    top10_tri_freqs4 = [[] for i in range(len(corpus_trigram))]
    for i, corp in enumerate(corpus_trigram):
        if i==0:
            temp3 = tfidf_trigram_model[corp]
            wd = int(xml_papersw[i][10])
            ####################################
            temp_top_ori = sorted(temp3, key=lambda x: x[1], reverse=True)
            temp_top_wrds_ori = [dictionary_trigram.get(x[0]) for x in temp_top_ori]
            top_trigram = [' '.join(re.findall(r'[\w\-]+\_[\w\-]+[\_[\w\-]+]*', word)) for word in temp_top_wrds_ori]
            while ("" in top_trigram):
                top_trigram.remove("")
            temp4_top10words = [(dictionary_trigram.get(x[0]), x[1]) for x in temp_top_ori]
            m = 0
            if wd == 1:
                for m, n in temp4_top10words:
                    if m in top_trigram:
                        temp5 = m.split('_')
                        temp6 = ''
                        for ii, tex in enumerate(temp5):  # Rejoining the trigrams with '_' again
                            temp6 = temp6 + token_word_dict.get(temp5[ii]) + ' '
                            # print(temp6)
                        top10_tri_words_original[i].append(temp6)
                        top10_tri_freqs[i].append(n)
                        # print(m,n, temp6)
                    else:
                        a = []
                        a.append(token_word_dict.get(m))
                        a = [x for x in a if x != None]
                        tagged = nltk.pos_tag(a)
                        for word, pos in tagged:
                            if pos != 'JJ' and not (len(token_word_dict.get(m)) <= 3 and token_word_dict.get(m).islower()):
                                top10_tri_words_original[i].append(token_word_dict.get(m))
                                top10_tri_freqs[i].append(n)
                    if (top10_tri_words_original[i].count(None) != 0):
                        m = top10_tri_words_original[i].index(None)
                        top10_tri_words_original[i] = list(filter(None, top10_tri_words_original[i]))
                        del top10_tri_freqs[m]
                    delete_repeat_max(top10_tri_words_original[i][:10], repeat_wrd[i], repeat_len[i])
                    if len(repeat_wrd[i]) != 0:
                        delete_repeat(top10_tri_words_original[i], top10_tri_freqs[i], repeat_wrd[i], repeat_len[i])
                    top10_tri_words_original3[i] = top10_tri_words_original[i][:10]
                    # top10_tri_words_original[i] = top10_tri_words_original[i][:10]
                    top10_tri_freqs3[i] = top10_tri_freqs[i][:10]
                    # top10_tri_freqs[i] = top10_tri_freqs[i][:10]
            else:
                for m, n in temp4_top10words:
                    if m in top_trigram:
                        temp5 = m.split('_')
                        temp6 = ''
                        for ii, tex in enumerate(temp5):  # Rejoining the trigrams with '_' again
                            temp6 = temp6 + token_word_dict.get(temp5[ii]) + ' '
                            # print(temp6)
                        top10_tri_words_original2[i].append(temp6)
                        top10_tri_freqs2[i].append(n)
                        # print(m,n, temp6)
                    else:
                        a = []
                        a.append(token_word_dict.get(m))
                        a = [x for x in a if x != None]
                        tagged = nltk.pos_tag(a)
                        for word, pos in tagged:
                            if pos != 'JJ' and not (len(token_word_dict.get(m)) <= 3 and token_word_dict.get(m).islower()):
                                top10_tri_words_original2[i].append(token_word_dict.get(m))
                                top10_tri_freqs2[i].append(n)
                    if (top10_tri_words_original2[i].count(None) != 0):
                        m = top10_tri_words_original2[i].index(None)
                        top10_tri_words_original2[i] = list(filter(None, top10_tri_words_original2[i]))
                        del top10_tri_freqs2[m]
                    delete_repeat_max(top10_tri_words_original2[i][:10], repeat_wrd[i], repeat_len[i])
                    if repeat_wrd[i] != 0:
                        delete_repeat(top10_tri_words_original2[i], top10_tri_freqs2[i], repeat_wrd[i], repeat_len[i])
                    top10_tri_words_original4[i] = top10_tri_words_original2[i][:10]
                    # top10_tri_words_original2[i] = top10_tri_words_original2[i][:10]
                    top10_tri_freqs4[i] = top10_tri_freqs2[i][:10]
                    # top10_tri_freqs2[i] = top10_tri_freqs2[i][:10]
                ##################################

        ### Plotting top 10 trigrams ###
        # for i in range(len(corpus_trigram)):
        #     txt=""
        #     wd = int(xml_papersw[i][10])
        #     if wd==0:
        #         list_fre=top10_tri_freqs4[i]
        #         list_wor = top10_tri_words_original4[i]
        #         dic = dict(zip(list_wor, list_fre))
        #         w = wordcloud.WordCloud(background_color="white")  # 把词云当做一个对象
        #         w.generate_from_frequencies(dic)
        #         w.to_file(resultPath + '\/' + f'The Short Article Trigram_figure_WorldCloud {xml_papersw[i][:-5]}.png')
        #
        #     if wd == 1:
        #         list_fre=top10_tri_freqs3[i]
        #         list_wor = top10_tri_words_original3[i]
        #         dic = dict(zip(list_wor, list_fre))
        #         w = wordcloud.WordCloud(background_color="white")  # 把词云当做一个对象
        #         w.generate_from_frequencies(dic)
        #         w.to_file(resultPath + '\/' + f'The Regular Article Trigram_figure_WorldCloud {xml_papersw[i][:-5]}.png')
        #
        # i=0
        # random.sample(range(0, len(xml_papers)), 30):
    for i in range(len(corpus_trigram)):
        # plt.figure(figsize=(24, 22))  # width:20, height:3
        # plt.barh(top10_tri_words_original[i], top10_tri_freqs[`i])
        wd = int(xml_papersw[i][10])
        if wd == 0:
            # plt.bar(top10_tri_words_original4[i], top10_tri_freqs4[i])
            # plt.title(f'The Short Article Top 10 trigrams "weighted"  for {article_Item[i]}')
            # plt.xticks(rotation=45, fontsize=11)
            # # Saving the figures in result path:
            # plt.savefig(os.path.join( f'{article_Item[i][:-4]}'), bbox_inches="tight")
            # plt.close()
            if article_Item[i] == article_Name:
                print("Short" + article_Item[i])
                print(top10_tri_words_original4[i])
                print("")

        if wd == 1:
            # plt.bar(top10_tri_words_original3[i], top10_tri_freqs3[i])
            # plt.title(f'The Regular Article Top 10 trigrams "weighted"  for {article_Item[i]}')
            # plt.xticks(rotation=45, fontsize=11)
            #
            # # Saving the figures in result path:
            # plt.savefig(os.path.join( f'{article_Item[i][:-4]}'), bbox_inches="tight")
            # plt.close()
            if article_Item[i] == article_Name:
                for num, abv in enumerate(top10_tri_words_original3[i]):
                    top10_tri_words_original3[i][num] = find_full_name(dict_doc_artiName[article_Item[i]], abv)
                print("Regular " + article_Item[i])
                print(top10_tri_words_original3[i])
                print("")


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
def return_MinMainString(text,str,pattern_First_Positions):
    for i in range(len(text)):
        if text[i] == str[0]:
            pattern_First_Positions.append(i)
    pattern_First_Positions=pattern_First_Positions[::-1]

    for num,position in  enumerate(pattern_First_Positions):
        match_Len=0
        i=0
        for x in range(position, len(text)):
            if text[x]==str[i]:
                match_Len+=1
                i += 1
        if match_Len==len(str):
            return num
    return -1
def return_MainString(text,str):
    lower_phrase = text.lower()
    match_Len=0
    i=0
    for x in range(0, len(lower_phrase)):
        if lower_phrase[x]==str[i]:
            match_Len+=1
            i += 1
            if match_Len==len(str):
                return 1
    return -1

def get_remaining_sentences(phrase, k,abv):

    phrase = re.sub('/', ' ', phrase)
    phrase = re.sub('-', ' ', phrase)
    abv = re.sub('-', '', abv)
    lower_phrase = phrase.lower()
    words = phrase.split(' ')
    lower_words=lower_phrase.split(' ')
    count = 0
    start_index = -1

    for i, word in enumerate(lower_words):
        if word.startswith(abv[0]):
            count += 1
            if count == k:
                start_index = i + 1
                break

    if start_index != -1:
        remaining_sentence = ' '.join(words[start_index-1:])
        return remaining_sentence
    else:
        return ""

class Solution:
    # 获取next数组
    def get_next(self, T):
        i = 0
        j = -1
        next_val = [-1] * len(T)
        while i < len(T) - 1:
            if j == -1 or T[i] == T[j]:
                i += 1
                j += 1
                # next_val[i] = j
                if i < len(T) and T[i] != T[j]:
                    next_val[i] = j
                else:
                    next_val[i] = next_val[j]
            else:
                j = next_val[j]
        return next_val
    # KMP算法
    def kmp(self, S, T):
        i = 0
        j = 0
        next = self.get_next(T)
        while i < len(S) and j < len(T):
            if j == -1 or S[i] == T[j]:
                i += 1
                j += 1
            else:
                j = next[j]
        if j == len(T):
            return i - j
        else:
            return -1
def merge_sublist_to_word(words, k, length):
    sublist = words[k:k + length]  # 获取从索引k开始，长度为length的子列表
    merged_word = ' '.join(sublist)  # 将子列表中的单词合并成一个单词

    return merged_word

def find_abbreviation(text, abbreviation):

    text_lower = text.lower()
    abbreviation_lower = abbreviation.lower()
    # 查找缩写在文本中的位置
    index = text_lower.find("(" + abbreviation_lower)
    real_abbreviation = ""
    if index != -1:
        while (text[index] != ")"):
            real_abbreviation += (text[index])
            index += 1

    if index != -1:
        return real_abbreviation
    else:
        return 0

def find_full_name(text, abbreviation):

    if any(char.isupper() for char in abbreviation):
        # 通过正则表达式匹配MINJ所在的短语
        length=len(abbreviation)
        match = re.search(r'\b(\w+\W+){0,%d}\(+%s\b' % (length*2, abbreviation), text)
        if match:
            print("yes")
        else:
            if find_abbreviation(text,abbreviation)!=0:
                abbreviation=find_abbreviation(text,abbreviation).replace("(","")
                match = re.search(r'\b(\w+\W+){0,%d}\(+%s\b' % (length * 2, abbreviation), text)
                length = len(abbreviation)
            else:
                return abbreviation

        # 获取匹配到的短语，并将其中的括号和MINJ去掉
        if match:
            phrase = match.group(0).replace('('+abbreviation, '').strip()

            clean_phrase = re.sub('-', ' ', phrase)
            clean_phrase = re.sub('/', ' ', clean_phrase)
            clean_phrase = re.sub(r'[^\w\s]', '', clean_phrase)
            # 获取短语中的前四个单词
            words = clean_phrase.split()[:length*2]
            initials = ''.join(word[0] for word in words)
            pattern=abbreviation.lower()
            s = Solution()
            start_situation=s.kmp(initials.lower(), pattern)
            if start_situation==-1:
                part2_words = clean_phrase.split(" ")
                initials = ''.join(word[0].lower() for word in part2_words)
                pattern_First_Positions = []
                num = return_MinMainString(initials, pattern, pattern_First_Positions)
                if num != -1:
                    orignal_Position = len(pattern_First_Positions) - num
                    return get_remaining_sentences(phrase, orignal_Position,pattern).replace("(","")+" ("+abbreviation+")"
                #第三种情况
                else:
                    pattern_First_Positions=pattern_First_Positions[::-1]
                    for num,position in enumerate(pattern_First_Positions):
                        orignal_Position = len(pattern_First_Positions)-num
                        less_FullName=get_remaining_sentences(phrase, orignal_Position,pattern)
                        less_FullNameString = ''.join(less_FullName).replace(" ","")
                        if return_MainString(less_FullNameString,pattern.replace("-",""))==1:
                            return less_FullName.replace("(","")+" ("+abbreviation+")"
                        n=0
            merged_word = merge_sublist_to_word(words, start_situation, length)
            return merged_word.replace("(","")+" ("+abbreviation+")"
        return abbreviation
    return abbreviation

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


def delete_repeat(txt,txt1,repeat_wrd,repeat_len):
    for i,wrd in enumerate(txt):
        if len(repeat_wrd)!=0:
            for k,wrd2 in enumerate(repeat_wrd):
                if wrd.find(wrd2)!=-1:
                    length=len(wrd)
                    if length<repeat_len[k]:
                        if txt.count(wrd)!=0:
                            txt[i]=""
                            txt1[i]=""
    while "" in txt1:
        txt1.remove("")
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
dict_doc_artiName=dict(zip(xml_papers,docs_list))
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
token_word_dict_pure = {}
tags = set(['FW', 'NN', 'NNS', 'NNP', 'NNPS'])
pure_Nworld=[]
for doc in data:
    data_new = []

    data_new=((doc).split(" "))
    tagged = nltk.pos_tag(data_new)
    data_new1 = []
    data_new2 = []
    for word, pos in tagged:
        if pos != 'MD':
            data_new1.append(word)
        if (pos in tags) and word!='COVID-19' and word!='SARS2019-CoV-2' and word!='Covid‐19'and word!='SARS2019-CoV':
            data_new2.append(word)

    var = ' '.join(data_new1)
    var2 = ' '.join(data_new2)
    mywords.append(preprocess(var)[0])
    pure_Nworld.append(preprocess(var2)[0])
    token_word_dict.update(preprocess(var)[1])
    token_word_dict_pure.update(preprocess(var2)[1])
            # print(preprocess(doc)[1])
# Removing words with frequency < 2:
# for sub in mywords:
#     sub[:] = [ele for ele in sub if sub.count(ele) > 1]
# token_word_dict = [x for x in token_word_dict if x != None]
# Building the bigram models
def remove_stopwrd(txt):
    result = []
    for wrd in txt:
        temp = wrd.split('_')
        if not any(ele in stpwrd for ele in temp):
            result.append(wrd)
    return result

pure_mywords_no_stopwrd = [[] for i in range(len(pure_Nworld))]
pure_mywords_no_stopwrd = [remove_stopwrd(lis) for lis in pure_Nworld]
pure_new_mywords_no_stopwrd = [" ".join(sub_list) for sub_list in pure_mywords_no_stopwrd]
vectorizer = CountVectorizer()
tf = vectorizer.fit_transform(pure_new_mywords_no_stopwrd)


def find_k(docs, min_k=1, max_k=40, min_df=2):
    # min_df 词语最少出现在2个文档中
    import matplotlib.pyplot as plt
    scores = []
    for k in range(min_k, max_k):
        # seed随机种子，保证在大邓这里运行结果与你运行的结果一样
        mdl = tp.LDAModel(min_df=min_df, k=k, seed=555)
        for words in docs:
            if words:
                mdl.add_doc(words)
        mdl.train(20)
        coh = tp.coherence.Coherence(mdl)
        scores.append(coh.get_score())

    # x = list(range(min_k, max_k - 1))  # 区间最右侧的值。注意：不能大于max_k
    # print(x)
    # print()
    plt.plot(range(min_k, max_k), scores)
    plt.xlabel("number of topics")
    plt.ylabel("coherence")
    plt.show()


# find_k(docs=pure_mywords_no_stopwrd, min_k=1, max_k=60, min_df=2)
#训练过程
md1=tp.HDPModel(alpha=0.8,seed=0)
for line in pure_Nworld:
    if line:
        md1.add_doc(line)
for i in range(0,100,10):
    md1.train(i)


for k in range(md1.k):
    print('Top 7 words of topic #{}'.format(k))
    print(md1.get_topic_words(k, top_n = 7))
#训练过程
dictionary_gram = gensim.corpora.Dictionary(pure_mywords_no_stopwrd)

corpus_gram= [dictionary_gram.doc2bow(text) for text in pure_mywords_no_stopwrd]
list_arti=[]
topic_term_dists = np.stack([md1.get_topic_word_dist(k) for k in range(md1.k)])
doc_topic_dists = np.stack([doc.get_topic_dist() for doc in md1.docs])
doc_topic_dists /= doc_topic_dists.sum(axis=1, keepdims=True)
doc_lengths = np.array([len(doc.words) for doc in md1.docs])
vocab = list(md1.used_vocabs)
term_frequency = md1.used_vocab_freq
i=0

# 使用布尔索引和列表推导式找到小于0.1的位置
result = [np.where(row < 0.1)[0].tolist() for row in doc_topic_dists]
print(result)

for doc in md1.docs:
    list_arti.append(np.argmax(doc.get_topic_dist()))
# Group indices of the same elements in a dictionary
index_dict=[]
topic_dict=[]
artcle_dict=[]
for j in range(md1.k):
    new_list=[]
    for i,item in enumerate(list_arti):
        if int(item) == j:
            topic_dict.append("Topic"+str(int(item)+1))
            new_list.append(xml_papers[i])
    artcle_dict.extend(new_list)
    index_dict.append(new_list)


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
mywords5_no_stopwrd = [[] for i in range(len(mywords5))]

mywords5_no_stopwrd = [remove_stopwrd(lis) for lis in mywords5]
docs_Final = dict.fromkeys(xml_papers)
positions = [xml_papers.index(item) for item in xd]
docs_Final = dict.fromkeys(xml_papers)
for k, article in enumerate(xml_papers):
    docs_Final[xml_papers[k]] =mywords5_no_stopwrd[k]
for i, item in enumerate(xd):
    article_Name=item
    mywords_no_stopwrd = []
    article_Name_List = []
    article_Name_List.append(item)
    for j ,item1 in enumerate(result[positions[i]]):
        article_Name_List+=index_dict[item1]

    result_Word = []
    for article_key in article_Name_List:
        if article_key in docs_Final:
            article_data = docs_Final[article_key]
            result_Word.append(article_data)
    tfidfProcess(result_Word,article_Name_List,article_Name)
    # print(mywords_no_stopwrd)

i=0
# 训 练