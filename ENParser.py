import re
import pickle
import emoji
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from statistics import mean

lemmatizer= WordNetLemmatizer()

name=lambda a : re.sub('[^a-zA-Z]','',emoji.demojize(a))

with open('emoji_model.pickle', 'rb') as data:
    emoji_net=pickle.load(data)  

def split_and_clean(sentance):
    text_part=re.sub('[^a-zA-Z]', ' ', sentance)
    lemmatized_words=[lemmatizer.lemmatize(words.lower(),pos="v") for words in text_part.split()]
    lemmatized_words=[lemmatizer.lemmatize(words.lower(),pos="n") for words in lemmatized_words]
    removed_stopped=[word for word in lemmatized_words if not word in set(stopwords.words('english'))]
    text_part=" ".join(removed_stopped)
    emoji_part=''.join([x for x in sentance if (x in emoji.UNICODE_EMOJI) or (x==' ')]).split()
    return text_part,emoji_part

similarity_matrix_list=list()
def decode_emoji(emojis,context):
    buffer_meanings_list=list()
    buffer_meanings=context
    reference_node=emoji_net
    
    for emoji_char in  emojis:
        if reference_node.has_child(name(emoji_char)):
            reference_node=reference_node.find_child(name(emoji_char))
        else:
            similarity_matrix=reference_node.find_similarity(buffer_meanings)
            similarity_matrix_list.append(similarity_matrix)
            buffer_meanings=highest_mean_similarity(similarity_matrix)
            buffer_meanings_list.append(buffer_meanings)
            reference_node=emoji_net.find_child(name(emoji_char)) 
            
            
    similarity_matrix=reference_node.find_similarity(buffer_meanings)
    similarity_matrix_list.append(similarity_matrix)
    buffer_meanings=highest_mean_similarity(similarity_matrix)
    buffer_meanings_list.append(buffer_meanings)
    
    return buffer_meanings_list


def highest_mean_similarity(matrix):
    avg_similarity=list()
    for i in range(len(matrix)):
        avg_similarity.append(mean(matrix.iloc[i].to_list()))
    
    pos=avg_similarity.index(max(avg_similarity))  
    return matrix.index[pos]
    

def create_partial_statement(comment):
    context,emojis = split_and_clean(comment)
    partial_statemnt=context.split()
    for emoji_seq in emojis:
        partial_statemnt+=decode_emoji(emoji_seq,' '.join(partial_statemnt))
    return context,emojis,' '.join(partial_statemnt)

