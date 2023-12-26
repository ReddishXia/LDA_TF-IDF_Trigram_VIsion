# -------------计算困惑度-------------------------------------
import codecs
import gensim
from gensim import corpora, models
import matplotlib.pyplot as plt
import matplotlib
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer

fp = codecs.open('test03.txt', 'r', encoding='utf-8')
data = fp.read()
fp.close()

# 以下几行将train列表的每个元素都生成一个列表 形成列表嵌套
train0 = data.split(" ")
train = []
for i in range(len(train0)):
    train1 = []
    train1.append(train0[i])
    train.append(train1)

dictionary = corpora.Dictionary(train)  # 构建 document-term matrix
corpus = [dictionary.doc2bow(text) for text in train]
Lda = gensim.models.ldamodel.LdaModel


def perplexity(num_topics):
    ldamodel = Lda(corpus, num_topics=num_topics, id2word=dictionary, passes=50)  # passes为迭代次数，次数越多越精准
    print(ldamodel.print_topics(num_topics=num_topics, num_words=7))  # num_words为每个主题下的词语数量
    print(ldamodel.log_perplexity(corpus))
    return ldamodel.log_perplexity(corpus)


# 绘制困惑度折线图
x = range(1, 10)  # 主题范围数量
y = [perplexity(i) for i in x]
plt.plot(x, y)
plt.xlabel('主题数目')
plt.ylabel('困惑度大小')
plt.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False
plt.title('主题-困惑度变化情况')
plt.show()

# ----------------计算一致性-------------------------------
# 计算coherence

from gensim.models.coherencemodel import CoherenceModel


def coherence(num_topics):
    ldamodel = Lda(corpus, num_topics=num_topics, id2word=dictionary, passes=30, random_state=1)
    # print(ldamodel.print_topics(num_topics=num_topics, num_words=7))
    ldacm = CoherenceModel(model=ldamodel, texts=train, dictionary=dictionary, coherence='c_v')
    # print(ldacm.get_coherence())
    return ldacm.get_coherence()


x = range(1, 10)
# z = [perplexity(i) for i in x]  #如果想用困惑度就选这个
y = [coherence(i) for i in x]
plt.plot(x, y)
plt.xlabel('主题数目')
plt.ylabel('一致性大小')
plt.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False
plt.title('主题-一致性变化情况')
plt.show()
