# -*- coding: utf-8 -*-
import pickle
import pandas as pd
import re
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from joblib import Parallel, delayed


path="TweetDataset"
df=pd.read_excel(path+"/Dataset.xlsx")

category_codes = {
    'sexual': 0,
    'offensive': 1,
    'Normal': 2
}


dataset=df[['Comment','Category']].copy()

#initializing
dataset['Review Lower']=dataset['Comment'].str.lower()
dataset['Review Words']=dataset['Comment']
dataset['Lemmatized Words']=dataset['Comment']
dataset['Stopped Removed']=dataset['Comment']


#lemmatization
lemmatizer= WordNetLemmatizer()


#Custom Stopwords
stopwordslist=['ain',"ain't",'www',
 'aren',
 "aren't",
 'don',
 "don't",
 'isn',
 "isn't",
 'no',
 'nor',
 'not',
 'wasn',
 "wasn't",
 'weren',
 "weren't",
 'i',
 "i'm"
 'me',
 'my',
 'don',
 "don't",
 'myself',
 'we',
 'our',
 'ours',
 'ourselves',
 "you're",
 "you've",
 "you'll",
 "you'd",
 'your',
 'yours',
 'yourself',
 'yourselves',
 'he',
 'him',
 'his',
 'himself',
 'she',
 "she's",
 'her',
 'hers',
 'herself',
 'it',
 "it's",
 'its',
 'itself',
 'they',
 'them',
 'their',
 'theirs',
 'themselves',
 'what',
 'which',
 'who',
 'whom',
 'this',
 'that',
 "that'll",
 'these',
 'those',
 'am',
 'is',
 'are',
 'was',
 'were',
 'be',
 'been',
 'being',
 'have',
 'has',
 'had',
 'having',
 'do',
 'does',
 'did',
 'doing',
 'a',
 'an',
 'the',
 'and',
 'but',
 'if',
 'or',
 'because',
 'as',
 'until',
 'while',
 'of',
 'at',
 'by',
 'for',
 'with',
 'about',
 'against',
 'between',
 'into',
 'through',
 'during',
 'before',
 'after',
 'above',
 'below',
 'to',
 'from',
 'up',
 'down',
 'in',
 'out',
 'on',
 'off',
 'over',
 'under',
 'again',
 'further',
 'then',
 'once',
 'here',
 'there',
 'when',
 'where',
 'why',
 'how',
 'all',
 'any',
 'both',
 'each',
 'few',
 'more',
 'most',
 'other',
 'some',
 'such',
 'only',
 'own',
 'same',
 'so',
 'than',
 'too',
 'very',
 's',
 't',
 'can',
 'will',
 'just',
 'should',
 "should've",
 'now',
 'd',
 'll',
 'm',
 'o',
 're',
 've',
 'y',
 'couldn',
 "couldn't",
 'hadn',
 "hadn't",
 'hasn',
 "hasn't",
 'haven',
 "haven't",
 'ma',
 'didn',
 "didn't",
 'doesn',
 "doesn't",
 'mightn',
 "mightn't",
 'mustn',
 "mustn't",
 'needn',
 "needn't",
 'shan',
 "shan't",
 'won',
 "won't",
 'wouldn',
 "wouldn't",
 'shouldn',
 "shouldn't"]




inputs = range(len(dataset))
def processInput(i):
    revised = re.sub(r"http\S+", "", dataset['Review Lower'][i])
    revised = re.sub("@[\w\d]+| *RT *|#[\w\d]+", "", revised)
    revised = re.sub("^\s*$| *# *", "", revised)
    review=re.sub('[^a-zA-Z]', ' ', revised)
    lemmatized_words=[lemmatizer.lemmatize(words,pos="v") for words in review.split()]
    lemmatized_words=[lemmatizer.lemmatize(words,pos="n") for words in lemmatized_words]
    removed_stopped=[word  for word in lemmatized_words if not word in set(stopwordslist)]
    lemmatized=" ".join(lemmatized_words)
    stopped=" ".join(removed_stopped)
    return [review,lemmatized,stopped]
    
content=pd.DataFrame()
content=content.append(Parallel(n_jobs=-1,prefer="threads",verbose=100)(delayed(processInput)(i) for i in inputs))

content.reset_index(drop=True, inplace=True)

dataset['Review Words']=content[0]
dataset['Lemmatized Words']=content[1]
dataset['Stopped Removed']=content[2]


dataset=dataset[['Comment','Category','Lemmatized Words','Stopped Removed']]


tfidf = TfidfVectorizer(encoding='utf-8',
                        ngram_range=(1,2),
                        stop_words=None,
                        lowercase=False,
                        min_df=20,
                        max_features=1000,
                        norm='l2',
                        sublinear_tf=True)



X_train, X_test, y_train, y_test = train_test_split(dataset['Stopped Removed'], 
                                                    dataset['Category'], 
                                                    test_size=0.2, 
                                                    random_state=42)

X_train=tfidf.fit_transform(X_train).toarray()
X_test=tfidf.transform(X_test).toarray()

featured_words=tfidf.get_feature_names()

#chi2 test for goodness
from sklearn.feature_selection import chi2
import numpy as np

for Product, category_id in sorted(category_codes.items()):
    features_chi2 = chi2(X_train, y_train == category_id)
    indices = np.argsort(features_chi2[0])
    feature_names = np.array(tfidf.get_feature_names())[indices]
    unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
    bigrams = [v for v in feature_names if len(v.split(' ')) == 2]
    print("# '{}' category:".format(Product))
    print("  . Most correlated unigrams:\n. {}".format('\n. '.join(unigrams[-5:])))
    print("  . Most correlated bigrams:\n. {}".format('\n. '.join(bigrams[-3:])))
    print("")


# Applying PCA
from sklearn.decomposition import PCA
pca = PCA(n_components = 750)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
explained_variance = pca.explained_variance_ratio_

#Plotting the Cumulative Summation of the Explained Variance
import matplotlib.pyplot as plt
import numpy as np
plt.figure()
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Variance (%)') #for each component
plt.title('Twitter Dataset Explained Variance')
plt.show()


# Dataset
with open(path+'/TwitterFinalDataset.pickle', 'wb') as output:
    pickle.dump(df, output)  
# X_train
with open(path+'/X_train.pickle', 'wb') as output:
    pickle.dump(X_train, output)  
# X_test    
with open(path+'/X_test.pickle', 'wb') as output:
    pickle.dump(X_test, output)
# y_train
with open(path+'/y_train.pickle', 'wb') as output:
    pickle.dump(y_train, output)
# y_test
with open(path+'/y_test.pickle', 'wb') as output:
    pickle.dump(y_test, output)
# TF-IDF object
with open(path+'/tfidf.pickle', 'wb') as output:
    pickle.dump(tfidf, output)
# PCA object
with open(path+'/PCA.pickle', 'wb') as output:
    pickle.dump(pca, output)
