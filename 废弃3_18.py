import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

# 下载nltk中必要的数据


# 输入多个英文文本
texts = [
    "The quick brown fox jumped over the lazy dog.",
    "The dog barked at the fox, but the fox kept running.",
    "The brown fox is quick and the dog is lazy."
]

# 将文本转换为小写
texts = [text.lower() for text in texts]

# 分句、分词、去除停用词、词形还原
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
processed_texts = []
for text in texts:
    sentences = sent_tokenize(text)
    for sentence in sentences:
        tokens = word_tokenize(sentence)
        tokens = [token for token in tokens if not token in stop_words]
        tokens = [lemmatizer.lemmatize(token) for token in tokens]
        processed_texts.append(' '.join(tokens))

# 计算TF-IDF值
vectorizer = TfidfVectorizer()
tf_idf_matrix = vectorizer.fit_transform(processed_texts)

# 输出TF-IDF值最高的前n个关键词
n = 3
feature_names = vectorizer.get_feature_names_out()
for i in range(len(texts)):
    text = texts[i]
    tf_idf_scores = tf_idf_matrix[i*len(sentences):(i+1)*len(sentences), :].mean(axis=0)
    sorted_indices = tf_idf_scores.argsort()[::-1][:n]
    keywords = [feature_names[j] for j in sorted_indices]
    print(f"Text {i+1}: {text.strip()}")
    print(f"Keywords: {', '.join(keywords)}\n")
