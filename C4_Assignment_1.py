
# coding: utf-8

# ---
# 
# _You are currently looking at **version 1.1** of this notebook. To download notebooks and datafiles, as well as get help on Jupyter notebooks in the Coursera platform, visit the [Jupyter Notebook FAQ](https://www.coursera.org/learn/python-text-mining/resources/d9pwm) course resource._
# 
# ---

# # Assignment 1
# 
# In this assignment, you'll be working with messy medical data and using regex to extract relevant infromation from the data. 
# 
# Each line of the `dates.txt` file corresponds to a medical note. Each note has a date that needs to be extracted, but each date is encoded in one of many formats.
# 
# The goal of this assignment is to correctly identify all of the different date variants encoded in this dataset and to properly normalize and sort the dates. 
# 
# Here is a list of some of the variants you might encounter in this dataset:
# * 04/20/2009; 04/20/09; 4/20/09; 4/3/09
# * Mar-20-2009; Mar 20, 2009; March 20, 2009;  Mar. 20, 2009; Mar 20 2009;
# * 20 Mar 2009; 20 March 2009; 20 Mar. 2009; 20 March, 2009
# * Mar 20th, 2009; Mar 21st, 2009; Mar 22nd, 2009
# * Feb 2009; Sep 2009; Oct 2010
# * 6/2008; 12/2009
# * 2009; 2010
# 
# Once you have extracted these date patterns from the text, the next step is to sort them in ascending chronological order accoring to the following rules:
# * Assume all dates in xx/xx/xx format are mm/dd/yy
# * Assume all dates where year is encoded in only two digits are years from the 1900's (e.g. 1/5/89 is January 5th, 1989)
# * If the day is missing (e.g. 9/2009), assume it is the first day of the month (e.g. September 1, 2009).
# * If the month is missing (e.g. 2010), assume it is the first of January of that year (e.g. January 1, 2010).
# * Watch out for potential typos as this is a raw, real-life derived dataset.
# 
# With these rules in mind, find the correct date in each note and return a pandas Series in chronological order of the original Series' indices.
# 
# For example if the original series was this:
# 
#     0    1999
#     1    2010
#     2    1978
#     3    2015
#     4    1985
# 
# Your function should return this:
# 
#     0    2
#     1    4
#     2    0
#     3    1
#     4    3
# 
# Your score will be calculated using [Kendall's tau](https://en.wikipedia.org/wiki/Kendall_rank_correlation_coefficient), a correlation measure for ordinal data.
# 
# *This function should return a Series of length 500 and dtype int.*

# In[1]:


import pandas as pd
import numpy as np


doc = []
with open('dates.txt') as file:
    for line in file:
        doc.append(line)

df = pd.Series(doc)

#df.iloc[14]


# In[2]:


#def date_sorter():
    
# group and find the hours and minutes
#df.str.findall(r'(\d?\d)/?(\d?\d)/(\d?\d?\d\d)')

#df.str.findall(r'(\d?\d)?[/-](\d?\d)[/-](\d?\d?\d\d)')

#df_dates = df.str.findall(r'(\d?\d|Jan.*?|Feb.*?|Mar.*?|Apr.*?|May|Jun.*?|Jul.*?|Aug.*?|Sep.*?|Oct.*?|Nov.*?|Dec.*?)?[ . /-]\
#(\d?\d|Jan.*?|Feb.*?|Mar.*?|Apr.*?|May|Jun.*?|Jul.*?|Aug.*?|Sep.*?|Oct.*?|Nov.*?|Dec.*?)[ . , /-](\d?\d?\d\d)')

#df_dates = df.str.findall(r'(\d?\d|Jan.*?|Feb.*?|Mar.*?|Apr.*?|May|Jun.*?|Jul.*?|Aug.*?|Sep.*?|Oct.*?|Nov.*?|Dec.*?)?[ ./-]\
#(\d?\d|Jan.*?|Feb.*?|Mar.*?|Apr.*?|May|Jun.*?|Jul.*?|Aug.*?|Sep.*?|Oct.*?|Nov.*?|Dec.*?)[ .,/-][\s]?(\d?\d?\d\d)')

#df_dates = df.str.findall(r'[ ./-]?(\d?\d|Jan.*?|Feb.*?|Mar.*?|Apr.*?|May|Jun.*?|Jul.*?|Aug.*?|Sep.*?|Oct.*?|Nov.*?|Dec.*?)[ ./-]\
#[\s]?(\d?\d|Jan.*?|Feb.*?|Mar.*?|Apr.*?|May|Jun.*?|Jul.*?|Aug.*?|Sep.*?|Oct.*?|Nov.*?|Dec.*?)[ .,/-][\s]?(\d?\d?\d\d)')

from datetime import datetime
from pandas import DataFrame
import re
from dateutil import parser
import pandas as pd
import numpy as np


doc = []
with open('dates.txt') as file:
    for line in file:
        doc.append(line)

df = pd.Series(doc)

df1_dates = df.str.findall(r'[ ./-]?(\d?\d|Jan.*?|Feb.*?|Mar|March|Apr.*?|May|Jun.*?|Jul|July|Aug|August|Sep.*?|Oct.*?|Nov.*?|Dec|December)[ ./-][\s]?(\d?\d|Jan.*?|Feb.*?|Mar|March|Apr.*?|May|Jun.*?|Jul|July|Aug|August|Sep.*?|Oct.*?|Nov.*?|Dec|December|)[ .,/-][\s]?(\d?\d?\d\d)')


df2_dates = df.str.findall(r'[^0-9]?[\s]?(\d?\d|Jan.*?|Feb.*?|Mar|March|Apr.*?|May|Jun.*?|Jul|July|Aug|August|Sep.*?|Oct.*?|Nov.*?|Dec|December|Decemeber)[ .,/-][ ]?((?:19|20)\d\d)')

df3_dates = df.str.findall(r'[\s]?((?:19|20)\d\d)')

#df_dates= df_dates[df_dates!=np.nan]
#type(df_dates)
#df2_dates
#df1_dates[[126,135,196,200,204,206,214,239,242,251,256,455,459,472,476]]
#type(df1_dates[1])
#df1_dates = [np.nan for i in df1_dates if len(i)==0]

#df1_merged = df1_dates.combine_first(df2_dates) 

df1_merged = pd.concat([df1_dates, df2_dates], axis=1)


#len(df1_merged.iloc[228,0])

#df1_dates[0:50][0]

#len(df1_dates[50])


#df1_merged[250:300]

#df2_merged = df1_merged.combine_first(df3_dates) 

#df2_merged

#df2_dates[490][0]

for i in range(len(df1_merged)):
    if len(df1_merged.iloc[i,0])==0:
        df1_merged.iloc[i,0] = df1_merged.iloc[i,1]

df1_merged.columns = ['zero','one']

#df1_merged

#df1_merged[200:230]

df2_merged = pd.concat([df1_merged, df3_dates], axis=1)

#df2_merged

for i in range(len(df2_merged)):
    if len(df2_merged.iloc[i,0])==0:
        df2_merged.iloc[i,0] = df2_merged.iloc[i,2]

#type(df2_merged.iloc[200,1])

#df2_merged = df2_merged[0]

#df2_merged[350:400]

df_unsorted = df2_merged['zero']

#df_unsorted[72] = df_unsorted[72][0]


#del df_sorted['level_O']

#df_final['index'].tail(25)

#print(df2_merged)

B=[1,2,3]

for i in range(len(df_unsorted)):
    df_unsorted[i] = np.asarray(df_unsorted[i])
    A=np.array_str(df_unsorted[i])
    A = A.replace('[','').replace(']','').replace('"','').replace("'",'')
    A=A.split()
    #print(len(A),A,type(A))
    if len(A)==3:
        B = np.vstack((B,A))
    if len(A)==2:
        B = np.vstack((B,np.hstack((str(1),A))))
    if len(A)==1:
        B = np.vstack((B,np.hstack((str(1),str(1),A))))

B1 = B

df_presorted = np.delete(B, np.where(B == [1, 2, 3]), axis=0)

df_presorted = df_presorted[1:]

#df_presorted[227]

C=[]
for i in range(len(df_presorted)):
    A = '-'.join(df_presorted[i][0:3])
    #print(A)
    B = list(A.split("-")) 
    #print(i,B)
    #C = np.vstack((C,B))
 #   print(i,C)
    C.append(B)

#type(C[0][0])
#type(B[0])

#print(len(df_unsorted))

C[297] = ['1', 'Jan', '1993']
C[312] = ['1', 'Dec', '1978']
C[391] = ['1', '5', '2000'] 
#instead of 67-5-2000
C[489] = ['1', '1', '2007'] 
#instead of 1-69-2007

s="-"
for i in range(len(df_presorted)):
    #print(i)
    C[i] = s.join(C[i])
    #print(i)
    parser.parse(C[i])

#len(df_presorted)

df_sorted = pd.to_datetime(C)

#df_sorted


df_sorted = pd.DataFrame({ 'date':df_sorted.values})

#df_sorted

df_sorted = df_sorted.sort_values(by='date',ascending=True)

#df_sorted


df_final = df_sorted.reset_index()

#df_final

del df_sorted['date']

#df_final['index']


    #return df_final['index']



#datetime.strptime(C[0])
#df_presorted[313][2]

#parser.parse(df_presorted[19][0])

#s=''
#for i in range(len(df_presorted)):
    #print(i)
 #   A = ''.join(df_presorted[i])
  #  s = s.append(A)
    #np.vstack(A)
    


#for i in range(len(df_presorted)):
#    df_sorted[i] = df_presorted[i].tostring()

#df_sorted




#return df_presorted
#df_unsorted[72][1]



#A
#parser.parse(my_string)

#A[1]
#type(df_presorted[256])

#df_presorted = df_unsorted


        
#df_presorted

#parser.parse(df_unsorted[1])

#df_unsorted[487]

#df1_dates[[126,135,196,200,204,206,214,239,242,251,256,455,459,472,476]]
#return pd.Series(data=[df_final['index']], dtype="int32")

#date_sorter()


# In[3]:


for i in range(len(df_unsorted)):
        df_unsorted[i] = df_unsorted[i][0]
        if len(df_unsorted[i])==2:
            month = pd.to_numeric(df_unsorted[i][0], errors='coerce')
            if month>12:
                df_unsorted[i]=df_unsorted[i][1]
        if len(df_unsorted[i])==3:
            day = pd.to_numeric(df_unsorted[i][0], errors='coerce')
            if day>31:
                df_unsorted[i]=df_unsorted[i][1:]

    #df_presorted=[]
for i in range(len(df_unsorted)):
    df_unsorted[i] = np.asarray(df_unsorted[i])

#df_unsorted[455]

df_presorted = df_unsorted

#type(df_presorted[454][0])

for i in range(len(df_presorted)):
    length_ = len(np.atleast_1d(df_presorted[i]))
    #print(i,length_)
    #if length_:
    if length_ == 2:
        df_presorted[i] = np.append([1],df_presorted[i])       
    #else:
     #   length_=1
    if length_ == 1:
        df_presorted[i] = np.append([1, 1],df_presorted[i]) 

#type(df_presorted[455])
#df_presorted[299]

df_presorted[298][1] = 'Jan'
df_presorted[313][1] = 'Dec'

s="-"
for i in range(len(df_presorted)):
    #print(i)
    df_presorted[i] = s.join(df_presorted[i])
    parser.parse(df_presorted[i])

df_sorted = pd.to_datetime(df_presorted)

#df_sorted[50:100]

df_sorted = pd.DataFrame({'index':df_sorted.index, 'date':df_sorted.values})

#df_sorted.sort('date')
#type(df_sorted['date'][455])

df_sorted = df_sorted.sort_values(by='date',ascending=True)
del df_sorted['date']

df_final = df_sorted.reset_index()

#df_final['index']

#df_presorted




    


# In[ ]:






# In[25]:



def date_sorter():
    
    regex1 = '(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})'
    regex2 = '((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[\S]*[+\s]\d{1,2}[,]{0,1}[+\s]\d{4})'
    regex3 = '(\d{1,2}[+\s](?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[\S]*[+\s]\d{4})'
    regex4 = '((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[\S]*[+\s]\d{4})'
    regex5 = '(\d{1,2}[/-][1|2]\d{3})'
    regex6 = '([1|2]\d{3})'
    full_regex = '(%s|%s|%s|%s|%s|%s)' %(regex1, regex2, regex3, regex4, regex5, regex6)
    parsed_date = df.str.extract(full_regex)
    parsed_date1 = parsed_date.iloc[:,0].str.replace('Janaury', 'January').str.replace('Decemeber', 'December')
    parsed_date2 = pd.Series(pd.to_datetime(parsed_date1))
    parsed_date3 = parsed_date2.sort_values(ascending=True).index
    values = pd.Series(parsed_date3.values)
    return values

date_sorter()


# In[ ]:




