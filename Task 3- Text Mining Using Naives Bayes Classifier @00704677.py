#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install numpy pandas matplotlib seaborn wordcloud nltk


# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import re
from  wordcloud import wordcloud
import nltk
nltk.download(['stopwords',
               'punkt',
               'wordnet',
               'omw-1.4',
               'vader_lexicon'])


# In[3]:


import sklearn as sk


# In[4]:


simple_text = 'This isn\'t a real text, this is an example text...Notice this contains punctuations!!'


# In[5]:


stop_words =nltk.corpus.stopwords.words('english')
print(stop_words)


# In[6]:


tokenizer =nltk.tokenize.RegexpTokenizer('[a-zA-Z0-9\']+')
tokenized_document =tokenizer.tokenize(simple_text)
print(tokenized_document)


# In[7]:


# we can also remove stopwords using list comprehension
cleaned_tokens =[word.lower() for word in tokenized_document if word.lower() not in stop_words]
print(cleaned_tokens)


# In[8]:


#Explore lemmatization vs stemming

lemmatizer = nltk.stem.WordNetLemmatizer()
stemmer =nltk.stem.PorterStemmer()

words =['cacti','sings','hopped', 'rocks','better','easily']
pos =['n','v','v','n','a','r']

lemmatized_words =[lemmatizer.lemmatize(words[i], pos=pos[i]) for i in range(6)]
stemmed_words= [stemmer.stem(word) for word in words]

print("Lemmatized words: ",  lemmatized_words)
print("Stemmed words: ", stemmed_words)


# In[9]:


#Now carry out stemming on our example setence
stemmed_text = [stemmer.stem(word) for word in cleaned_tokens]

print(stemmed_text)


# In[10]:


#lets now create a function to apply all of our data preprocesing steps which we can then use on a corpus

def preprocess_text(text):
    tokenized_document = nltk.tokenize.RegexpTokenizer('[a-zA-Z0-9\']+').tokenize(text)  # Tokenize
    cleaned_tokens =[word.lower() for word in tokenized_document if word.lower() not in stop_words] #Remove
    stemmed_text =[nltk.stem.PorterStemmer().stem(word)for word in cleaned_tokens] #Stemming
    return stemmed_text


# In[11]:


British_Airways = pd.read_csv('British_Airway_Review.csv')


# In[12]:


British_Airways


# In[13]:


print("\n All Data Labels")
print(British_Airways.groupby("recommended").count())


# In[14]:


British_Airways['reviews'] = British_Airways['reviews'].apply(preprocess_text)


# In[15]:


British_Airways.head()


# In[21]:


from sklearn.feature_extraction.text import CountVectorizer
vectorizer =CountVectorizer()
X =vectorizer.fit_transform(British_Airways['reviews'].map(' '.join))
X =pd.DataFrame(X.toarray())
X.head()


# In[17]:


from sklearn.model_selection import train_test_split


y =British_Airways['recommended']

X_train, X_test, y_train,y_test =train_test_split(
X,y, train_size=0.8,test_size=0.2,random_state=90)


# In[18]:


from sklearn.naive_bayes import MultinomialNB

model =MultinomialNB()
model.fit(X_train, y_train)

MultinomialNB()


# In[19]:


y_pred =model.predict(X_test)
from sklearn import metrics
acc=metrics.accuracy_score(y_test,y_pred)
print('accuracy:%.2f\n\n'%(acc))
cm=metrics.confusion_matrix(y_test,y_pred)
print('Confusion Matrix:')
print(cm,'\n\n')
print('-------------------------------------------------')
result=metrics.classification_report(y_test,y_pred)
print('Classification Report:\n')
print(result)


# In[20]:


British_Airways

