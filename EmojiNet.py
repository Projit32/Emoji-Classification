from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
from nltk.corpus import wordnet
import spacy 
from joblib import Parallel, delayed
import multiprocessing
import pandas as pd

lemmatizer=WordNetLemmatizer()

negation=['ain',"ain't",
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
 "weren't"]

stopwordslist=['i',
 'me',
 'my',
 'don',
 "don't",
 'myself',
 'we',
 'our',
 'ours',
 'ourselves',
 'you',
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
 'want'
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

nlp = spacy.load('en_core_web_lg') 


def checkSimilarity(words):
    tokens = nlp(words) 
    """for token in tokens: 
        print(token.text, token.has_vector, token.vector_norm, token.is_oov)""" 
    token1, token2 = tokens[0], tokens[1] 
    return token1.similarity(token2)*100

def findOpposite(word):
    antonyms = [] 
    for syn in wordnet.synsets(word): 
        for l in syn.lemmas(): 
            if l.antonyms(): 
                antonyms.append(l.antonyms()[0].name())
    
    if len(antonyms)>0:
        return antonyms[0]
    else:
        return word

def clean(sentence):
        sentence=sentence.lower()
        sentence=re.sub('[^a-zA-Z]',' ',sentence)
        sentence=' '.join([lemmatizer.lemmatize(word,pos="v") for word in sentence.split() if word not in set(stopwordslist)])
        return sentence

def changeIfNegetive(sentence):
    sentence=clean(sentence)
    words=sentence.split()
    indexList=[i for i, word in enumerate(words) if word.lower() in negation]
    final=[]
    for i in range(len(words)): 
        if i-1 in indexList:
            final.append(findOpposite(words[i]))
        elif words[i] not in negation:
            final.append(words[i])
            
    return final


def binarySearch(arr, target):
    start = 0
    end = len(arr) - 1
    while start <= end:
        middle = (start + end)// 2
        midpoint = arr[middle].name
        if midpoint > target:
            end = middle - 1
        elif midpoint < target:
            start = middle + 1
        else:
            return True,middle
    return False,-1

class Node:
        
    def __init__(self,emoji_name,emoji):
        self.name=emoji_name
        self.emoji=emoji
        self.parent=None
        self.child_list=list()
        self.bag_of_words=set()
    
    def name_comparator(self,child_node):
        return child_node.name
    
    def add_child(self,child):
        self.child_list.sort(key = self.name_comparator)
        present,position =binarySearch(self.child_list,child.name)
        if not present:
            self.child_list.append(child)
            child.parent=self
            return child
        else:
            return self.child_list[position]
    
    def equals(self, node):
        return self.name == node.name
    
    def find_similarity(self,context):
        context = clean(context)
        context = context.split()
        similarity_matrix=pd.DataFrame()
        for means in self.bag_of_words:
            values=list()
            for words in context:
                print(words+' '+means)
                values.append(checkSimilarity(words+' '+means))
            values.insert(0,means) 
            print(values)
            similarity_matrix=similarity_matrix.append([values])
 
        similarity_matrix.set_index(0,inplace=True)   
        similarity_matrix.columns=context
        
        return similarity_matrix.copy(deep=True)
        
    def find_child(self, child_name):
        return self.child_list[binarySearch(self.child_list,child_name)[1]]
    
    def has_child(self, child_name):
        return binarySearch(self.child_list,child_name)[0]
    
    def print_children(self):
        children=[child.emoji for child in self.child_list]
        print(children)
        
    def add_to_bag(self,description):
        word_list=changeIfNegetive(description)
        for word in word_list:
            self.bag_of_words.add(word)