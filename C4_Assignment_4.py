
# coding: utf-8

# ---
# 
# _You are currently looking at **version 1.0** of this notebook. To download notebooks and datafiles, as well as get help on Jupyter notebooks in the Coursera platform, visit the [Jupyter Notebook FAQ](https://www.coursera.org/learn/python-text-mining/resources/d9pwm) course resource._
# 
# ---

# # Assignment 4 - Document Similarity & Topic Modelling

# ## Part 1 - Document Similarity
# 
# For the first part of this assignment, you will complete the functions `doc_to_synsets` and `similarity_score` which will be used by `document_path_similarity` to find the path similarity between two documents.
# 
# The following functions are provided:
# * **`convert_tag:`** converts the tag given by `nltk.pos_tag` to a tag used by `wordnet.synsets`. You will need to use this function in `doc_to_synsets`.
# * **`document_path_similarity:`** computes the symmetrical path similarity between two documents by finding the synsets in each document using `doc_to_synsets`, then computing similarities using `similarity_score`.
# 
# You will need to finish writing the following functions:
# * **`doc_to_synsets:`** returns a list of synsets in document. This function should first tokenize and part of speech tag the document using `nltk.word_tokenize` and `nltk.pos_tag`. Then it should find each tokens corresponding synset using `wn.synsets(token, wordnet_tag)`. The first synset match should be used. If there is no match, that token is skipped.
# * **`similarity_score:`** returns the normalized similarity score of a list of synsets (s1) onto a second list of synsets (s2). For each synset in s1, find the synset in s2 with the largest similarity value. Sum all of the largest similarity values together and normalize this value by dividing it by the number of largest similarity values found. Be careful with data types, which should be floats. Missing values should be ignored.
# 
# Once `doc_to_synsets` and `similarity_score` have been completed, submit to the autograder which will run `test_document_path_similarity` to test that these functions are running correctly. 
# 
# *Do not modify the functions `convert_tag`, `document_path_similarity`, and `test_document_path_similarity`.*

# In[23]:


import numpy as np
import nltk
from nltk.corpus import wordnet as wn
import pandas as pd
nltk.download("book")
from nltk.book import *


def convert_tag(tag):
    """Convert the tag given by nltk.pos_tag to the tag used by wordnet.synsets"""
    
    #tag_dict =  {'D': 'n', 'I': 'n', 'P': 'n', 'N': 'n', 'J': 'a', 'R': 'r', 'V': 'v'}
    tag_dict = {'N': 'n', 'J': 'a', 'R': 'r', 'V': 'v'}
    try:
        return tag_dict[tag[0]]
    except KeyError:
        return None


def doc_to_synsets(doc):
    """
    Returns a list of synsets in document.

    Tokenizes and tags the words in the document doc.
    Then finds the first synset for each word/tag combination.
    If a synset is not found for that combination it is skipped.

    Args:
        doc: string to be converted

    Returns:
        list of synsets

    Example:
        doc_to_synsets('Fish are nvqjp friends.')
        Out: [Synset('fish.n.01'), Synset('be.v.01'), Synset('friend.n.01')]
    """
    

    token = nltk.word_tokenize(doc)
    #wn.synsets(token)[0]
    pos = nltk.pos_tag(token)
    tags = [x[1] for x in pos]

    wntag = [convert_tag(x) for x in tags]

    ans = list(zip(token,wntag))

    sets = [wn.synsets(x,y) for x,y in ans]

    list_synset=[]
    for i in range(len(sets)):
        if sets[i]==[]:
            continue
        else:
            list_synset.append(sets[i][0])
        
    
    return list_synset


def similarity_score(s1, s2):
    """
    Calculate the normalized similarity score of s1 onto s2

    For each synset in s1, finds the synset in s2 with the largest similarity value.
    Sum of all of the largest similarity values and normalize this value by dividing it by the
    number of largest similarity values found.

    Args:
        s1, s2: list of synsets from doc_to_synsets

    Returns:
        normalized similarity score of s1 onto s2

    Example:
        synsets1 = doc_to_synsets('I like cats')
        synsets2 = doc_to_synsets('I like dogs')
        similarity_score(synsets1, synsets2)
        Out: 0.73333333333333339
    """
    max_sims=[]

    for i in s2:
        sims=[]
        for j in s1:
            d = wordnet.path_similarity(i, j)
            sims.append((d))
        scores = [i for i in sims if i is not None]
        if len(scores)>0:
            max_sims.append(max(scores))
            #(max([i for i in sims if i is not None]))
    if len(max_sims)==0:
        res=0
    else:
        res = sum(max_sims)/len(max_sims)
        
    return res


def document_path_similarity(doc1, doc2):
    """Finds the symmetrical similarity between doc1 and doc2"""

    synsets1 = doc_to_synsets(doc1)
    synsets2 = doc_to_synsets(doc2)

    return (similarity_score(synsets1, synsets2) + similarity_score(synsets2, synsets1)) / 2


# In[22]:


doc = 'I like cats.'
doc2 = 'I like dogs'


token = nltk.word_tokenize(doc)
#wn.synsets(token)[0]
pos = nltk.pos_tag(token)
tags = [x[1] for x in pos]

wntag = [convert_tag(x) for x in tags]

ans = list(zip(token,wntag))

sets1 = [wn.synsets(x,y) for x,y in ans]

A=[]
for i in range(len(sets1)):
    if sets1[i]==[]:
        continue
    else:
        A.append(sets1[i][0])
        
token2 = nltk.word_tokenize(doc2)
#wn.synsets(token)[0]
pos2 = nltk.pos_tag(token2)
tags2 = [x[1] for x in pos2]

wntag2 = [convert_tag(x) for x in tags2]

ans2 = list(zip(token2,wntag2))

sets2 = [wn.synsets(x,y) for x,y in ans2]

B=[]
for i in range(len(sets2)):
    if sets2[i]==[]:
        continue
    else:
        B.append(sets2[i][0])
        

    
s1 = doc_to_synsets('This is a function to test document_path_similarity')
s2 = doc_to_synsets('Use this function to see if your code in doc_to_synsets and similarity_score is correct!')
#s2[0]='Synset('use.v.01')'

max_sims=[]

for i in s1:
    sims=[]
    for j in s2:
        d = wordnet.path_similarity(i, j)
        sims.append((d))
    scores = [i for i in sims if i is not None]
    if len(scores)>0:
        max_sims.append(max(scores))
        #(max([i for i in sims if i is not None]))
if len(max_sims)==0:
    res=0
else:
    res = sum(max_sims)/len(max_sims)

    

#if len(scores) > 0:
#largest.append(max(scores))
    
    
scores,res,max_sims


# ### test_document_path_similarity
# 
# Use this function to check if doc_to_synsets and similarity_score are correct.
# 
# *This function should return the similarity score as a float.*

# In[24]:


def test_document_path_similarity():
    doc1 = 'This is a function to test document_path_similarity.'
    doc2 = 'Use this function to see if your code in doc_to_synsets and similarity_score is correct!'
    #doc1='I like cats'
    #doc2='I like dogs'
    return document_path_similarity(doc1, doc2)

test_document_path_similarity()


# In[4]:


doc1 = 'This is a function to test document_path_similarity.'


# <br>
# ___
# `paraphrases` is a DataFrame which contains the following columns: `Quality`, `D1`, and `D2`.
# 
# `Quality` is an indicator variable which indicates if the two documents `D1` and `D2` are paraphrases of one another (1 for paraphrase, 0 for not paraphrase).

# In[41]:


# Use this dataframe for questions most_similar_docs and label_accuracy
paraphrases = pd.read_csv('paraphrases.csv')
paraphrases.head()
#paraphrases.iloc[:,0]


# ___
# 
# ### most_similar_docs
# 
# Using `document_path_similarity`, find the pair of documents in paraphrases which has the maximum similarity score.
# 
# *This function should return a tuple `(D1, D2, similarity_score)`*

# In[39]:


def most_similar_docs():
    
    scores_label=[]
    for i in range(len(paraphrases)):
        k=document_path_similarity(paraphrases.iloc[i,1],paraphrases.iloc[i,2])
        scores_label.append(k)
        #print(i)
        max_value = max(scores_label)
        max_index = scores_label.index(max_value)
    return paraphrases.iloc[max_index,1],paraphrases.iloc[max_index,2],max_value

most_similar_docs()


# ### label_accuracy
# 
# Provide labels for the twenty pairs of documents by computing the similarity for each pair using `document_path_similarity`. Let the classifier rule be that if the score is greater than 0.75, label is paraphrase (1), else label is not paraphrase (0). Report accuracy of the classifier using scikit-learn's accuracy_score.
# 
# *This function should return a float.*

# In[49]:


def label_accuracy():
    from sklearn.metrics import accuracy_score

    scores_label=[]
    for i in range(len(paraphrases)):
        k=document_path_similarity(paraphrases.iloc[i,1],paraphrases.iloc[i,2])
        scores_label.append(k)
        #max_value = max(scores_label)
        #max_index = scores_label.index(max_value)
    paraphrases['Label'] = [1 if scores_label[i]>0.75 else 0 for i in range(len(scores_label))]

    return accuracy_score(paraphrases['Label'],paraphrases['Quality'])

label_accuracy()


# ## Part 2 - Topic Modelling
# 
# For the second part of this assignment, you will use Gensim's LDA (Latent Dirichlet Allocation) model to model topics in `newsgroup_data`. You will first need to finish the code in the cell below by using gensim.models.ldamodel.LdaModel constructor to estimate LDA model parameters on the corpus, and save to the variable `ldamodel`. Extract 10 topics using `corpus` and `id_map`, and with `passes=25` and `random_state=34`.

# In[2]:


import pickle
import gensim
from sklearn.feature_extraction.text import CountVectorizer

# Load the list of documents
with open('newsgroups', 'rb') as f:
    newsgroup_data = pickle.load(f)

# Use CountVectorizor to find three letter tokens, remove stop_words, 
# remove tokens that don't appear in at least 20 documents,
# remove tokens that appear in more than 20% of the documents
vect = CountVectorizer(min_df=20, max_df=0.2, stop_words='english', 
                       token_pattern='(?u)\\b\\w\\w\\w+\\b')
# Fit and transform
X = vect.fit_transform(newsgroup_data)

# Convert sparse matrix to gensim corpus.
corpus = gensim.matutils.Sparse2Corpus(X, documents_columns=False)

# Mapping from word IDs to words (To be used in LdaModel's id2word parameter)
id_map = dict((v, k) for k, v in vect.vocabulary_.items())

#id_map


# In[3]:


# Use the gensim.models.ldamodel.LdaModel constructor to estimate 
# LDA model parameters on the corpus, and save to the variable `ldamodel`

# Your code here:
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics = 10)
#id2word=dictionary, passes=15)


# ### lda_topics
# 
# Using `ldamodel`, find a list of the 10 topics and the most significant 10 words in each topic. This should be structured as a list of 10 tuples where each tuple takes on the form:
# 
# `(9, '0.068*"space" + 0.036*"nasa" + 0.021*"science" + 0.020*"edu" + 0.019*"data" + 0.017*"shuttle" + 0.015*"launch" + 0.015*"available" + 0.014*"center" + 0.014*"sci"')`
# 
# for example.
# 
# *This function should return a list of tuples.*

# In[4]:


def lda_topics():
    
    topics = ldamodel.print_topics(num_words=10)
    #for topic in topics:
     #   print(topic)
    
    return topics

lda_topics()


# ### topic_distribution
# 
# For the new document `new_doc`, find the topic distribution. Remember to use vect.transform on the the new doc, and Sparse2Corpus to convert the sparse matrix to gensim corpus.
# 
# *This function should return a list of tuples, where each tuple is `(#topic, probability)`*

# In[5]:


new_doc = ["\n\nIt's my understanding that the freezing will start to occur because of the\ngrowing distance of Pluto and Charon from the Sun, due to it's\nelliptical orbit. It is not due to shadowing effects. \n\n\nPluto can shadow Charon, and vice-versa.\n\nGeorge Krumins\n-- "]


# In[11]:


def topic_distribution():
    
    b = vect.transform(new_doc)
    # Convert sparse matrix to gensim corpus.
    new_doc_corpus = gensim.matutils.Sparse2Corpus(b, documents_columns=False)
    # get the topics (note: minimum_probability=0.01 is not needed)
    c = ldamodel.get_document_topics(new_doc_corpus, minimum_probability=0.01)
    return list(c)[0]

topic_distribution()


# ### topic_names
# 
# From the list of the following given topics, assign topic names to the topics you found. If none of these names best matches the topics you found, create a new 1-3 word "title" for the topic.
# 
# Topics: Health, Science, Automobiles, Politics, Government, Travel, Computers & IT, Sports, Business, Society & Lifestyle, Religion, Education.
# 
# *This function should return a list of 10 strings.*

# In[13]:


def topic_names():
    
    # Your Code Here
    
    return # Your Answer Here

