
# coding: utf-8

# ---
# 
# _You are currently looking at **version 1.1** of this notebook. To download notebooks and datafiles, as well as get help on Jupyter notebooks in the Coursera platform, visit the [Jupyter Notebook FAQ](https://www.coursera.org/learn/python-text-mining/resources/d9pwm) course resource._
# 
# ---

# # Assignment 3
# 
# In this assignment you will explore text message data and create models to predict if a message is spam or not. 

# In[1]:


import pandas as pd
import numpy as np

spam_data = pd.read_csv('spam.csv')

spam_data['target'] = np.where(spam_data['target']=='spam',1,0)
spam_data.head(10)
len(spam_data)


# In[2]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(spam_data['text'], 
                                                    spam_data['target'], 
                                                    random_state=0)

len(X_train)


# ### Question 1
# What percentage of the documents in `spam_data` are spam?
# 
# *This function should return a float, the percent value (i.e. $ratio * 100$).*

# In[3]:


def answer_one():
    
    
    return spam_data['target'].mean()*100

answer_one()


# In[ ]:





# ### Question 2
# 
# Fit the training data `X_train` using a Count Vectorizer with default parameters.
# 
# What is the longest token in the vocabulary?
# 
# *This function should return a string.*

# In[4]:


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
    
def answer_two():
    X_train, X_test, y_train, y_test = train_test_split(spam_data['text'], 
                                                    spam_data['target'], 
                                                    random_state=0)
    vect = CountVectorizer().fit(X_train)
    words = vect.get_feature_names()
    res = max(words, key = len) 

    return res

answer_two()


# In[ ]:





# ### Question 3
# 
# Fit and transform the training data `X_train` using a Count Vectorizer with default parameters.
# 
# Next, fit a fit a multinomial Naive Bayes classifier model with smoothing `alpha=0.1`. Find the area under the curve (AUC) score using the transformed test data.
# 
# *This function should return the AUC score as a float.*

# In[5]:


from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import roc_auc_score

    
def answer_three():
    vect = CountVectorizer().fit(X_train)
    # transform the documents in the training data to a document-term matrix
    X_train_vectorized = vect.transform(X_train)
    
    model = MultinomialNB(alpha=0.1)
    model.fit(X_train_vectorized, y_train)
    
    predictions = model.predict(vect.transform(X_test))
 
    return roc_auc_score(y_test, predictions)

answer_three()


# In[ ]:





# ### Question 4
# 
# Fit and transform the training data `X_train` using a Tfidf Vectorizer with default parameters.
# 
# What 20 features have the smallest tf-idf and what 20 have the largest tf-idf?
# 
# Put these features in a two series where each series is sorted by tf-idf value and then alphabetically by feature name. The index of the series should be the feature name, and the data should be the tf-idf.
# 
# The series of 20 features with smallest tf-idfs should be sorted smallest tfidf first, the list of 20 features with largest tf-idfs should be sorted largest first. 
# 
# *This function should return a tuple of two series
# `(smallest tf-idfs series, largest tf-idfs series)`.*

# In[6]:


from sklearn.feature_extraction.text import TfidfVectorizer

def answer_four():
    
    tfidf = TfidfVectorizer().fit(X_train)
    feature_names = np.array(tfidf.get_feature_names())
    
    X_train_tf = tfidf.transform(X_train)
    
    max_tf_idfs = X_train_tf.max(0).toarray()[0] # Get largest tfidf values across all documents.
    sorted_tf_idxs = max_tf_idfs.argsort() # Sorted indices
    sorted_tf_idfs = max_tf_idfs[sorted_tf_idxs] # Sorted TFIDF values
    
    # feature_names doesn't need to be sorted! You just access it with a list of sorted indices!
    smallest_tf_idfs = pd.Series(sorted_tf_idfs[:20], index=feature_names[sorted_tf_idxs[:20]])                    
    largest_tf_idfs = pd.Series(sorted_tf_idfs[-20:][::-1], index=feature_names[sorted_tf_idxs[-20:][::-1]])
    
    return (smallest_tf_idfs, largest_tf_idfs)
#df['Feature'] = df

#df.shape

#len(feature_names)
#sorted_
#feature_names[feature_index]
#df



    #return (feature_names[sorted_tfidf_index[:20]],feature_names[sorted_tfidf_index[-21:-1]]), X_train_vectorized.max(0).toarray()[0].argsort()

#answer_four()


# In[ ]:





# ### Question 5
# 
# Fit and transform the training data `X_train` using a Tfidf Vectorizer ignoring terms that have a document frequency strictly lower than **3**.
# 
# Then fit a multinomial Naive Bayes classifier model with smoothing `alpha=0.1` and compute the area under the curve (AUC) score using the transformed test data.
# 
# *This function should return the AUC score as a float.*

# In[7]:


def answer_five():
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics import roc_auc_score

    
    vect = TfidfVectorizer(min_df=3).fit(X_train)
    # transform the documents in the training data to a document-term matrix
    X_train_vectorized = vect.transform(X_train)
    
    model = MultinomialNB(alpha=0.1)
    model.fit(X_train_vectorized, y_train)
    
    predictions = model.predict(vect.transform(X_test))
 
    return roc_auc_score(y_test, predictions)

answer_five()
    
    
  


# In[ ]:





# ### Question 6
# 
# What is the average length of documents (number of characters) for not spam and spam documents?
# 
# *This function should return a tuple (average length not spam, average length spam).*

# In[8]:


def answer_six():
    
    import numpy as np
    df1 = spam_data[spam_data['target']==0]
    df2 = spam_data[spam_data['target']==1]

    #df1['text'][1]

    avg_not_spam = sum( map(len, df1['text']) ) / len(df1['text'])
    avg_spam = sum( map(len, df2['text']) ) / len(df2['text'])

    #sum( map(len, df1['text']) )
    return avg_not_spam,avg_spam

answer_six()


# In[9]:


answer_six()


# <br>
# <br>
# The following function has been provided to help you combine new features into the training data:

# In[10]:


def add_feature(X, feature_to_add):
    """
    Returns sparse feature matrix with added feature.
    feature_to_add can also be a list of features.
    """
    from scipy.sparse import csr_matrix, hstack
    return hstack([X, csr_matrix(feature_to_add).T], 'csr')


# ### Question 7
# 
# Fit and transform the training data X_train using a Tfidf Vectorizer ignoring terms that have a document frequency strictly lower than **5**.
# 
# Using this document-term matrix and an additional feature, **the length of document (number of characters)**, fit a Support Vector Classification model with regularization `C=10000`. Then compute the area under the curve (AUC) score using the transformed test data.
# 
# *This function should return the AUC score as a float.*

# In[11]:


from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score

def answer_seven():
    
    vect = TfidfVectorizer(min_df=5).fit(X_train)
    model = SVC(C=10000)
    # transform the documents in the training data to a document-term matrix
    X_train_vectorized = vect.transform(X_train)
    X_test_vectorized = vect.transform(X_test)

    X_train_vectorized_added = add_feature(X_train_vectorized, X_train.str.len())
    X_test_vectorized_added = add_feature(X_test_vectorized, X_test.str.len())
    model.fit(X_train_vectorized_added, y_train)

    prediction_scores = model.predict(X_test_vectorized_added)
    #prediction_scores
    return roc_auc_score(y_test, prediction_scores)

answer_seven()
    
    
   


# In[12]:


def count_digits(string):
    return sum(item.isdigit() for item in string)

def count_letters(string):
    return sum(item.isalpha() or item == '_' for item in string)


# ### Question 8
# 
# What is the average number of digits per document for not spam and spam documents?
# 
# *This function should return a tuple (average # digits not spam, average # digits spam).*

# In[13]:


def answer_eight():
    
    import numpy as np

    #numbers = sum(c.isdigit() for c in s)

    df1 = spam_data[spam_data['target']==0]
    df2 = spam_data[spam_data['target']==1]

    df1['counts'] = df1['text'].apply(count_digits)
    df2['counts'] = df2['text'].apply(count_digits)


    return df1['counts'].mean(),df2['counts'].mean()


    #return avg_digits_not_spam,avg_digits_spam

answer_eight()


# In[ ]:





# ### Question 9
# 
# Fit and transform the training data `X_train` using a Tfidf Vectorizer ignoring terms that have a document frequency strictly lower than **5** and using **word n-grams from n=1 to n=3** (unigrams, bigrams, and trigrams).
# 
# Using this document-term matrix and the following additional features:
# * the length of document (number of characters)
# * **number of digits per document**
# 
# fit a Logistic Regression model with regularization `C=100`. Then compute the area under the curve (AUC) score using the transformed test data.
# 
# *This function should return the AUC score as a float.*

# In[29]:


from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score

def answer_nine1():
    vect = TfidfVectorizer(min_df=5).fit(X_train)
    model = LogisticRegression(C=100,random_state=0)

    #X_train['counts'] = X_train.apply(count_digits)
    #X_test['counts'] = X_test.apply(count_digits)

    #print('',X_test)
    #print('',X_train.apply(count_digits).sum())

    # transform the documents in the training data to a document-term matrix
    X_train_vectorized = vect.transform(X_train)
    X_test_vectorized = vect.transform(X_test)

    #X_train_vectorized_added = add_feature(X_train_vectorized, X_train.str.len())
    X_train_vectorized_final = add_feature(X_train_vectorized, X_train.apply(count_digits))

    #X_test_vectorized_added = add_feature(X_test_vectorized, X_test.str.len())
    X_test_vectorized_final = add_feature(X_test_vectorized, X_test.apply(count_digits))
    model.fit(X_train_vectorized_final, y_train)

    prediction_scores = model.predict(X_test_vectorized_final)
    #prediction_scores[:,1]
    return float(roc_auc_score(y_test, prediction_scores)),X_test.apply(count_digits)
    
answer_nine1()


# In[28]:


from sklearn.linear_model import LogisticRegression

def answer_nine():
    
    dig_train = [sum(char.isnumeric() for char in x) for x in X_train]
    dig_test = [sum(char.isnumeric() for char in x) for x in X_test]
    
    tf = TfidfVectorizer(min_df = 5, ngram_range = (1,3)).fit(X_train)
    X_train_tf = tf.transform(X_train)
    X_test_tf = tf.transform(X_test)
    
    X_train_tf = add_feature(X_train_tf, dig_train)
    X_test_tf = add_feature(X_test_tf, dig_test)
    
    clf = LogisticRegression(C=100).fit(X_train_tf, y_train)
    pred = clf.predict(X_test_tf)
    
    return roc_auc_score(y_test, pred),dig_test

answer_nine()


# ### Question 10
# 
# What is the average number of non-word characters (anything other than a letter, digit or underscore) per document for not spam and spam documents?
# 
# *Hint: Use `\w` and `\W` character classes*
# 
# *This function should return a tuple (average # non-word characters not spam, average # non-word characters spam).*

# In[15]:


def answer_ten():
    df1 = spam_data[spam_data['target']==0]
    df2 = spam_data[spam_data['target']==1]

    df1['digits'] = df1['text'].apply(count_digits)
    df2['digits'] = df2['text'].apply(count_digits)

    df1['letters'] = df1['text'].apply(count_letters)
    df2['letters'] = df2['text'].apply(count_letters)


    non_word_not_spam = df1['text'].str.len().sum() - df1['digits'].sum() - df1['letters'].sum()
    non_word_spam = df2['text'].str.len().sum() - df2['digits'].sum() - df2['letters'].sum()

    return non_word_not_spam/len(df1),non_word_spam/len(df2)
    
answer_ten()


# In[ ]:





# ### Question 11
# 
# Fit and transform the training data X_train using a Count Vectorizer ignoring terms that have a document frequency strictly lower than **5** and using **character n-grams from n=2 to n=5.**
# 
# To tell Count Vectorizer to use character n-grams pass in `analyzer='char_wb'` which creates character n-grams only from text inside word boundaries. This should make the model more robust to spelling mistakes.
# 
# Using this document-term matrix and the following additional features:
# * the length of document (number of characters)
# * number of digits per document
# * **number of non-word characters (anything other than a letter, digit or underscore.)**
# 
# fit a Logistic Regression model with regularization C=100. Then compute the area under the curve (AUC) score using the transformed test data.
# 
# Also **find the 10 smallest and 10 largest coefficients from the model** and return them along with the AUC score in a tuple.
# 
# The list of 10 smallest coefficients should be sorted smallest first, the list of 10 largest coefficients should be sorted largest first.
# 
# The three features that were added to the document term matrix should have the following names should they appear in the list of coefficients:
# ['length_of_doc', 'digit_count', 'non_word_char_count']
# 
# *This function should return a tuple `(AUC score as a float, smallest coefs list, largest coefs list)`.*

# In[35]:


def answer_eleven():
    vect = CountVectorizer(min_df=5,ngram_range=(2,5),analyzer='char_wb').fit(X_train)
    model = LogisticRegression(C=100)


    # transform the documents in the training data to a document-term matrix
    X_train_vectorized = vect.transform(X_train)
    X_test_vectorized = vect.transform(X_test)
    
    dig_train = [sum(char.isnumeric() for char in x) for x in X_train]
    dig_test = [sum(char.isnumeric() for char in x) for x in X_test]
    
    let_train = [sum(char.isnumeric() or char == '_' for char in x) for x in X_train]
    let_test = [sum(char.isnumeric() or char == '_' for char in x) for x in X_test]

    non_word_train = X_train.str.len().sum() - dig_train - let_train
    non_word_test = X_test.str.len().sum() - dig_test - let_test

    #X_train_vectorized_added = add_feature(X_train_vectorized, X_train.str.len())
    X_train_vectorized_final = add_feature(X_train_vectorized, [X_train.str.len(),dig_train,non_word_train])

    #X_test_vectorized_added = add_feature(X_test_vectorized, X_test.str.len())
    X_test_vectorized_final = add_feature(X_test_vectorized,  [X_test.str.len(),dig_test,non_word_test])
    model.fit(X_train_vectorized_final, y_train)

    prediction_scores = model.predict(X_test_vectorized_final)
    
    feature_names = np.array(vect.get_feature_names() + ['length_of_doc', 'digit_count', 'non_word_char_count'])

    sorted_coef_index = model.coef_[0].argsort() 
    #prediction_scores[:,1]
    return float(roc_auc_score(y_test, prediction_scores)), feature_names[sorted_coef_index[:10]], feature_names[sorted_coef_index[-11:-1]]
    
answer_eleven()


# In[17]:




