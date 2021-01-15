import pandas as pd
#import tensorflow as tf
#import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelBinarizer
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
import re,string,unicodedata
import nltk
df = pd.read_csv("columns/train1.csv")
print(df.describe())




#Removing the square brackets
def remove_between_square_brackets(text):
    return re.sub('\[[^]]*\]', '', text)

#Removing the noisy text
def denoise_text(text):

    text = remove_between_square_brackets(text)
    return text
#Apply function on review column
df['data']=df['data'].apply(denoise_text)

def remove_special_characters(text, remove_digits=True):
    pattern=r'[^a-zA-z0-9\s]'
    text=re.sub(pattern,'',text)
    return text
#Apply function on review column
df['data']=df['data'].apply(remove_special_characters)



#Stemming the text
def simple_stemmer(text):
    ps=nltk.porter.PorterStemmer()
    text= ' '.join([ps.stem(word) for word in text.split()])
    return text
#Apply function on review column
df['data']=df['data'].apply(simple_stemmer)

train_data = df['data'].to_numpy()[:60]
train_sentiment = df['labels'].to_numpy()[:60]
print(train_data)
print(train_sentiment)

test_data = df['data'].to_numpy()[60:]
test_sentiment = df['labels'].to_numpy()[60:]
print(train_data.shape,train_sentiment.shape)
print(test_data.shape,test_sentiment.shape)


cv=CountVectorizer(min_df=0,max_df=1,binary=False,ngram_range=(1,3))
#transformed train reviews
cv_train_reviews=cv.fit_transform(train_data)
#transformed test reviews
cv_test_reviews=cv.transform(test_data)
print('BOW_cv_train:',cv_train_reviews.shape)
print('BOW_cv_test:',cv_test_reviews.shape)



#Tfidf vectorizer
tv=TfidfVectorizer(min_df=0,max_df=1,use_idf=True,ngram_range=(1,3))
#transformed train reviews
tv_train_reviews=tv.fit_transform(train_data)
#transformed test reviews
tv_test_reviews=tv.transform(test_data)
print('Tfidf_train:',tv_train_reviews.shape)
print('Tfidf_test:',tv_test_reviews.shape)

#labeling the sentient data
lb=LabelBinarizer()
sentiment_data=lb.fit_transform(df['labels'])
print(sentiment_data.shape)
train_sentiments=sentiment_data[:60]
test_sentiments=sentiment_data[60:]
print(train_sentiments)
print(test_sentiments)

lr=LogisticRegression(penalty='l2',max_iter=500,C=1,random_state=20)
#Fitting the model for Bag of words
lr_bow=lr.fit(cv_train_reviews,train_sentiments)
print(lr_bow)
#Fitting the model for tfidf features
lr_tfidf=lr.fit(tv_train_reviews,train_sentiments)
print(lr_tfidf)

lr_bow_predict=lr.predict(cv_test_reviews)
print(lr_bow_predict)
##Predicting the model for tfidf features
lr_tfidf_predict=lr.predict(tv_test_reviews)
print(lr_tfidf_predict)

#Accuracy score for bag of words
lr_bow_score=accuracy_score(test_sentiments,lr_bow_predict)
print("lr_bow_score :",lr_bow_score)
#Accuracy score for tfidf features
lr_tfidf_score=accuracy_score(test_sentiments,lr_tfidf_predict)
print("lr_tfidf_score :",lr_tfidf_score)










