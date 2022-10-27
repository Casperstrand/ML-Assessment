import pandas as pd
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import model_selection, svm

stopwords = nltk.corpus.stopwords.words('english')
lemmetizer = nltk.WordNetLemmatizer()
vectorizer = CountVectorizer()

def remove_stop_words(text):
    text = [w for w in text if w.lower() not in stopwords]
    return text

def lemmetize_words(word_list):
    lemmetized = [lemmetizer.lemmatize(w) for w in word_list]
    return lemmetized

def vectorize(row):
    for line in row:
        line = vectorizer.fit_transform(line)

df = pd.read_csv('news_articles.csv')
df = df.dropna()
df['text'] = df['text'].apply(nltk.word_tokenize)
df['text'] = df['text'].apply(remove_stop_words)
df['text'] = df['text'].apply(lemmetize_words)
df['text'] = df['text'].apply(vectorizer.fit_transform)

train_x, test_x, train_y, test_y = model_selection.train_test_split(df['text'], df['label'], test_size=0.2)

