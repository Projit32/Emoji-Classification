# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 18:31:54 2020

@author: proji
"""

import ENParser as enp
import pickle
from tensorflow.keras.models import load_model
import pandas as pd
import tensorflow

path="TweetDataset/"

classifier = load_model(path+'ANNTFV3.1.h5')
with open(path+"tfidf.pickle", 'rb') as data:
    vectorizer = pickle.load(data)
with open(path+"pca.pickle", 'rb') as data:
    dr = pickle.load(data)

# with open(path+"RandomForest.pickle", 'rb') as data:
#     classifier = pickle.load(data)
    

# features=vectorizer.get_feature_names()

tensorflow.keras.utils.plot_model(
    classifier, to_file='model.png', show_shapes=True, show_layer_names=True,
    rankdir='TB', expand_nested=True, dpi=150)


#read dataset
comments=[]
comments.append("I'll will chop you, cut you 🔪💀 and throw you")
comments.append("your comment was a complete 🐂💩🍆, 🔩😒 man... ")
comments.append("I'm gonna 🖕 you up you bitch")
comments.append("Let's have some 👉👌 tonight... or better 👅🍑💦")
comments.append("this was nice")
comments.append("Good morning! how are you today?")
comments.append('I can see that you\'re horny 😈😏')


import time
start_time = time.time()

#Partial Statement generator
partial_comment=[]
splits=pd.DataFrame()
for comment in comments:
    text,emoji,ps=enp.create_partial_statement(comment)
    partial_comment.append(ps)
    splits=splits.append([[comment,text,emoji]])
splits.columns=['Comment','Text part','Emoji Part']

    
#vectorizing and reducing statement
vectorized_comment=vectorizer.transform(partial_comment).toarray()
reduced_comment=dr.transform(vectorized_comment)

#predictions
predicts=classifier.predict_classes(reduced_comment)
values=classifier.predict(reduced_comment)

predictions=pd.DataFrame()
for i in range(len(predicts)):
    if predicts[i]==1:
        category='Offensive'
    elif predicts[i]==0:
        category='sexual'
    else:
        category="Neutral"
    predictions=predictions.append([[comments[i],category,values[i]]])

predictions.columns=['Comment','Category','Values']
predictions.reset_index(drop=True, inplace=True)

print("--- %s seconds ---" % (time.time() - start_time))