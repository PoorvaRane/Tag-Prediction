
# coding: utf-8

# In[2]:

import nltk 
import re, pprint
import urllib as url
import urllib2 as url2
from bs4 import BeautifulSoup
import xml.etree.ElementTree


# In[3]:

'''
This function is used to get the root of the xml tree.
Input- filename
Output- root of xml tree
'''
def getroot_xml(filename):
    root = xml.etree.ElementTree.parse('Posts_small.xml').getroot()
    return root


# In[ ]:

'''
This function reads the text in questions 
Input- root of xml tree
Output- Parsed question strings
'''
def get_questions(root):
    questions = []
    for row in root.findall('row'):
        body = row.get("Body")
        soup = BeautifulSoup(body)
        [s.extract() for s in soup('code')]
        question = soup.get_text()
        print question
        questions.append(question)
    return questions
