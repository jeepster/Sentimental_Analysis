import numpy as np
import nltk

from nltk.stem import WordNetLemmatizer # to convert words like jumps to jump thereby reduce vocabulary size
from sklearn.linear_model import LogisticRegression # using for classification
from bs4 import BeautifulSoup # for parsing XML files
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LinearRegression # using regression for classification

word_lemmatizer = WordNetLemmatizer()

stopwords = set([w.rstrip()  for w in open('sentimental_analysis/dataset/stopwords.txt')])

# loads the html data to a set using BS
positive_reviews = BeautifulSoup(open('sentimental_analysis/dataset/positive.review'))
negative_reviews = BeautifulSoup(open('sentimental_analysis/dataset/negative.review'))

#parses using keyword
positive_reviews = positive_reviews.findAll('review_text')
negative_reviews = negative_reviews.findAll('review_text')

#shuffling and equalizing the size of both review sets
np.random.shuffle(positive_reviews)
positive_reviews = positive_reviews[:len(negative_reviews)]


#custom tokenizer
def my_tokenizer(s):
    s = s.lower()
    tokens = nltk.tokenize.word_tokenize(s)
    tokens = [t for t in tokens if len(t)>2] # to remove words with length less than 2 as they dont have any meaning
    tokens = [word_lemmatizer.lemmatize(t) for t in tokens] 
    tokens = [t for t in tokens if t not in stopwords] # to remove stopwords from sentences
    return tokens
    
#building vocabulary of words in a dict
word_index_map = {}
index = 0               # vocabulary size

positive_tokenized = []
negative_tokenized = []

for review in positive_reviews:
    tokens = my_tokenizer(review.text)
    positive_tokenized.append(tokens)
    #every sentence in a review
    for token in tokens:
        if token not in word_index_map:
            word_index_map[token] = index
            index +=1

for review in negative_reviews:
    tokens = my_tokenizer(review.text)
    negative_tokenized.append(tokens)
    #every sentence in a review
    for token in tokens:
        if token not in word_index_map:
            word_index_map[token] = index
            index +=1


def tokenstovector(tokens,label):
    x = np.zeros(len(word_index_map) + 1)
    for t in tokens:
        i = word_index_map[t]
        x[i] +=1
    x = x/x.sum()
    x[-1] = label
    return x

N = len(positive_tokenized) + len(negative_tokenized)
data = np.zeros((N,len(word_index_map) + 1))
i = 0

for tokens in positive_tokenized:
    xy = tokenstovector(tokens,5) #Changed labels for linear regression on scale of 1 to 5 for negative & positive reviews respectively
    data[i,:] = xy
    i +=1
for tokens in negative_tokenized:
    xy = tokenstovector(tokens,1)
    data[i,:] = xy
    i+=1
    
np.random.shuffle(data)

X = data[:,:-1]
Y = data[:,-1]

Xtrain = X[:-100,]
Ytrain = Y[:-100,]
Xtest  = X[-100:,]
Ytest  = Y[-100:,]

model = LogisticRegression()
model.fit(Xtrain,Ytrain)
print('Classification rate is for LR : ',model.score(Xtest,Ytest))

model2 = AdaBoostClassifier()
model2.fit(Xtrain,Ytrain)
    print('Classification Rate by AdaBoost : ',model2.score(Xtest,Ytest))

model3 = LinearRegression()
model3.fit(Xtrain,Ytrain)
print('Regression Rate by LinearRegression : ',model3.score(Xtest,Ytest))

# prediction
X1 = "\nI bought this for easy transfer of pictures from my digital camera with SD memory card anywhere not home and sometimes from other peoples memory card (xD and memory stick)..\n\nFirst of all I was disappointed with the flimsy, plastic design and the size of it. But it would have been ok if it worked!..IT DOESNT READ my SD card. And as menetioned in other people's review hard to insert and take out the cards! I'm scared if the cards get scratch and ruined whenever i do it. I wish I have bought this after reading amazon reviews...it's useless now. I'm lost how I can get sd card work on this and if I do, scared of frequent use for the flimsy design\n"
X1tokens = my_tokenizer(X1)
X2 = tokenstovector(X1tokens,0)
X2 = X2[:11078] # reducing one column of labels
X2 = X2.reshape(1,11078) # reshaping into 2D vector
# print(model.predict(X2))
# print(model2.predict(X2))
print(model3.predict(X2))
    
threshold = 3
for word,index in word_index_map.items():
#     weight = model.coef_[0][index]
#     print(weight)
#     if weight>threshold or weight<-threshold:
#         print(word,weight)
#     weight2 = model2.coef_[0][index]
#     if weight2>threshold or weight2<-threshold:
#         print(word,index)
    weight2 = model3.coef_[index]
    if weight2>threshold or weight2<-threshold:
        print(word,weight2)
    

