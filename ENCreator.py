import pandas as pd
from EmojiNet import Node
import emoji
import re
import pickle

dataset=pd.read_excel('Emoji Dataset.xlsx')

name=lambda a : re.sub('[^a-zA-Z]','',emoji.demojize(a))
emoji_net=Node("Root","Root")



def create_tree(seq,nodeReference,desc):
    if name(seq[0]).find('skintone')==-1:
        print(name(seq[0]))
        nodeReference=nodeReference.add_child(Node(name(seq[0]),seq[0]))
    if len(seq)==1:
        nodeReference.add_to_bag(desc)
        print(nodeReference.bag_of_words)
        return
    else:
        create_tree(seq[1:],nodeReference,desc)
        

for i in  range(len(dataset)):
    emoji_set=dataset.Emoji[i]
    description=dataset.Meaning[i]
    print(emoji_set)
    create_tree(emoji_set.replace(" ", "") ,emoji_net,description)
    

with open('emoji_model.pickle', 'wb') as output:
    pickle.dump(emoji_net, output)