
# coding: utf-8

# In[55]:

from __future__ import division
import nltk 
import re
from bs4 import BeautifulSoup
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import lxml.html
from nltk.corpus import stopwords
from nltk.stem import *
import networkx as nx
import json
import pickle
from llda import LLDA
from bayesian_ic import BIC
from optparse import OptionParser


# In[56]:

'''
This function is used to get the root of the xml tree.
Input- filename
Output- root of xml tree
'''
def getroot_xml(filename):
    root = ET.parse(filename).getroot()
    return root


# In[57]:

'''
This function parses xml data
Input- root of xml tree
Output- Title, Question body, Title + Question, Tags
'''
def get_questions_tags(root):
    questions = []
    tags = []
    titles = []
    users = []
    ques_with_title = []
    for row in root.findall('row'):
        post = row.get("PostTypeId")
        user = row.get("OwnerUserId")
        post_type = BeautifulSoup(post, "lxml")
        if post_type.get_text() == "1" and user is not None:
            
            users.append(user)
            
            #Get the Questions
            body = row.get("Body")
            soup = BeautifulSoup(body, "lxml")
            [s.extract() for s in soup('code')] #To remove code from the question posted
            question_s = soup.get_text()
#             q_set = nltk.word_tokenize(question_s)
            q_set = question_s.split()
            question = nltk.Text(q_set)
            questions.append(question)
            
            #Get the Tags
            tag_list = row.get("Tags")
            tag_str = re.sub('[<>]', ' ', tag_list)
#             tag_set = nltk.word_tokenize(tag_str)
            tag_set = tag_str.split()        
            tag_text = nltk.Text(tag_set)
            tags.append(tag_text)

            #Get the Titles
            title_s = row.get("Title")
#             t_set = nltk.word_tokenize(title_s)
            t_set = title_s.split()
            title = nltk.Text(t_set)
            titles.append(title)
            
            q_with_t = title_s + " " + question_s
#             qt_set = nltk.word_tokenize(q_with_t)
            qt_set = q_with_t.split()
            ques_title = nltk.Text(qt_set)
            ques_with_title.append(ques_title)

    return questions, tags, titles, ques_with_title, users


# In[58]:

'''
This function removes numbers and special characters from question
'''
def number_removal(ques_list):
    q_with_t_list = []
    for ques in ques_list:
        q_t_text = ""
        for word in ques:
            characters = [".", ",", ":", "(", ")", "[", "]", "{", "}", "?", "'"]
            q_text = ''.join([i for i in word if not (i.isdigit() or [e for e in characters if e in i])])
            if q_text != '':
                q_t_text += q_text + " "
        qt_set = q_t_text.split()
#         qt_set = nltk.word_tokenize(q_t_text)
        title_ques = nltk.Text(qt_set)
        q_with_t_list.append(title_ques)
    return q_with_t_list


# In[59]:

'''
This function removes stopwords from the question body
'''
def remove_stopwords(q_list):
    q_with_t_list = []
    for text in q_list:
        stopwords = nltk.corpus.stopwords.words('english')
        st = ""
        for w in text:
            if w.lower() not in stopwords:
                st += w.lower() + " "
        w_set = st.split()
#         w_set = nltk.word_tokenize(st)
        ques_body = nltk.Text(w_set)
        q_with_t_list.append(ques_body)
    return q_with_t_list


# In[60]:

'''
This function performs stemming and coverts each word in the question to it's root word
'''
def stemming(q_list):
    stemmer = PorterStemmer()
    post = []
    for q in q_list:
        st = ""
        for word in q:
            st += stemmer.stem(word) + " "
#         w_set = nltk.word_tokenize(st)
        w_set = st.split()
        ques_body = nltk.Text(w_set)
        post.append(ques_body)
    return post


# In[61]:

'''
This function will do the final fixes. Last stage of pre-processing
'''
def fixing(text):
    post = []
    for tokens in text:
        for i, t in enumerate(tokens):
            if t == '#' and i > 0:
                left = tokens[:i-1]
                joined = [tokens[i - 1] + t]
                right = tokens[i + 1:]
                tokens = left + joined + right
        post.append(tokens)
    return post


# In[62]:

'''
Preprocessor component - Tokenisation, Number removal, Stop-word removal, Stemming
'''
def preprocessor(filename):
    root = getroot_xml(filename)
    questions, tags, titles, ques_with_title, users = get_questions_tags(root)
    ques_with_title_list = number_removal(ques_with_title)
    title_ques = remove_stopwords(ques_with_title_list)
    posts = stemming(title_ques)
    frame = pd.DataFrame({0 : titles,
                          1 : questions,
                          2 : ques_with_title,
                          3 : posts,
                          4 : tags,
                          5 : users})
    return frame


# In[103]:

'''
Frequentist Inference Component - POS Tagger
'''
def pos_tag(posts):
    pos_tags = []
    for post in posts:
        tag = []
        for word in post:
            te = nltk.pos_tag([word])
            t = te[0]
            if t[1].startswith('NN') or t[1].startswith('JJ'):
                tag.append(t)
        pos_tags.append(tag)
    with open('postags.json', 'w') as outfile:
        json.dump(pos_tags, outfile)
    return pos_tags


# In[104]:

'''
Write POS TAG output to file
'''
def write_to_file(filename, posts):
    o = open(filename,'w')
    pos = pos_tag(posts)
    o.write(str(pos))
    o.close()


# In[105]:

'''
Set of all possible tags
'''
def set_of_tags(tag_column, fileName):
    tags = []
    for row in tag_column:
        for t in row:
            if t not in tags:
                tags.append(t)
    f = open(fileName, "wb")
    pickle.dump(tags, f)
    f.close()
#     print("completed")
#     if return_flag:
    return tags


# In[106]:

'''
Reads the postag file
'''
def readfile(filename):
    f = open(filename,'r')
    software_object = []
    for line in f.readlines():
        print line


# In[107]:

'''
Creating post frame
'''
def create_post_frame(pos_tag_list, tag_column, fileName):
    posts = []
    tags = []
    for i in range(len(pos_tag_list)):
        post_row = []
        ptl = pos_tag_list[i]
        tc = tag_column[i]
        tags.append(tc)
        for p in ptl:
            post_row.append(p[0])
        posts.append(post_row)
    frame = pd.DataFrame({0 : posts,
                          1 : tags})
    f = open(fileName, "wb")
    pickle.dump(frame, f)
    f.close()
#     print("completed")
#     if return_flag:
    return frame


# In[108]:

'''
Creating post frame with userids
'''
def create_post_frame_user(pos_tag_list, tag_column, userids):
    posts = []
    tags = []
    users = []
    for i in range(len(pos_tag_list)):
        post_row = []
        ptl = pos_tag_list[i]
        tc = tag_column[i]
        uc = userids[i]
        users.append(uc)
        tags.append(tc)
        for p in ptl:
            post_row.append(p[0])
        posts.append(post_row)
    frame = pd.DataFrame({0 : posts,
                          1 : tags,
                          2 : users})
    return frame


# In[119]:

def create_weights(c, fileName):
    tags = set()
    for row in c[1]:
        tags.update(row)
    tags = list(tags)
    output = {}
    for i in xrange(len(tags)):
        for j in xrange(i+1, len(tags)):
            if (tags[i], tags[j]) not in output and (tags[j], tags[i]) not in output:
                pioj = 0
                pinj = 0
                for line in c[1]:
                    if tags[i] in line and tags[j] in line:
                        pinj += 1
                    if tags[i] in line or tags[j] in line:
                        pioj += 1
                output[(tags[i], tags[j])] = pinj/float(pioj)
    
    f = open(fileName, "wb")
    pickle.dump(output, f)
    f.close()
#     print("completed")
#     if return_flag:
    return output


# In[110]:

'''
Compute Probability - FREPOS
'''
def probability(so, tags, post_frame):
    s_prob_t = {}
    for tag in tags:
        s_prob = {}
        count = 0
        for word in so:
            word = word[0]
            s_prob[word] = 0
            for i in range(len(post_frame)):
                if word in post_frame[0][i] and tag in post_frame[1][i]:
                    s_prob[word] = 1
                    count += 1
                    break
        if count > 0:
            s_prob_t[tag] = count/len(s_prob)
    return s_prob_t


# In[111]:

'''
Spreading Activation Algorithm
'''
def fic(G, hops=2):
    main_nodes = [n for n in G.nodes() if G.node.get(n, 0)]
    for node in main_nodes:
        nodes = [(node, 0)]
        while nodes:
            n, k = nodes.pop(0)
            if k > hops:
                break
            adjNodes = G.neighbors(n)
            for adjNode in adjNodes:
#                 if adjNode in main_nodes:
#                     continue
                val = G.node[n].get('value', 0) * G.get_edge_data(n, adjNode)['weight']
                if val > G.node[adjNode].get('value', 0):
                    G.node[adjNode]['value'] = val
                nodes.append((adjNode, k + 1))
    return {k: v['value'] for k, v in G.nodes(data=True) if 'value' in v}


# In[112]:

import networkx as nx
def getGraph(probs, edge_wts):
    graph = nx.Graph()
    for node, value in probs.iteritems():
        graph.add_node(node, {'value': value})
    for edge, wt in edge_wts.iteritems():
        if wt:
            graph.add_edge(edge[0], edge[1], {'weight': wt})
    return graph


# In[113]:

def fic_output(tot_probs, edge_wts):
    output = []
    for prob in tot_probs:
        G = getGraph(prob, edge_wts)
        output.append(fic(G))
    return output


# In[114]:

def uhic(userids, tags):
    users = []
    user_tags = []
    for i in range(len(userids)):
        u_row = userids[i]
        t_row = tags[i]
        if u_row not in users:
            users.append(u_row)
            u_tag = {}
            for t in t_row:
                u_tag[t] = 1
            user_tags.append(u_tag)
        else:
            ind = users.index(u_row)
            p_t = user_tags[ind]
            for t in t_row:
                if t in p_t.keys():
                    p_t[t] += 1
                else:
                    p_t[t] = 1
    frame = pd.DataFrame({0 : users,
                          1 : user_tags})
    return frame


# In[115]:

def bic_output(dictionary, tot_so):
    output = []
    for so in tot_so:
        word_tag = {}
        for word in so:
            for k, v in dictionary.items():
                if word[0] in v:
                    word_tag[k] = v[word[0]]
        output.append(word_tag)
    return output


# In[116]:

def getTotalProbability(tot_so, tags, soNtags, fileName):
    probs = []
    for so in tot_so:
        try:
            probs.append(probability(so, tags, soNtags))
        except:
            continue
    f = open(fileName, "wb")
    pickle.dump(probs, f)
    f.close()
#     print("completed")
#     if return_flag:
    return probs


# In[125]:

pre = preprocessor('Posts_small.xml')
tot_so = pos_tag(pre[3])
tags = set_of_tags(pre[4], 'tags.pkl')
soNtags = create_post_frame(tot_so, pre[4], 'soNtags.pkl')
edge_wts = create_weights(soNtags, 'egdeWeights.pkl')


# In[126]:

tot_probs = getTotalProbability(tot_so, tags, soNtags, 'probabilities.pkl')


# In[ ]:

# tot_so = json.load(open('postags.json', 'r'))
# tags = pickle.load(open('tags.pkl', 'rb'))
# # edge_wts = pickle.load(open('edgeWeights.pkl', 'rb'))
# tot_probs = pickle.load(open('probabilities.pkl', 'rb'))
# soNtags = pickle.load(open('soNtags.pkl', 'rb'))


# In[128]:

diction = BIC(tags,pre[3],pre[4])


# In[129]:

tot_bic_op = bic_output(diction, tot_so)
tot_fic_op = fic_output(tot_probs, edge_wts)


# In[130]:

'''
Recall@k
'''
def evalCrit(predTags, corrTags):
    p = set(predTags)
    c = set(corrTags)
    output = len(p.intersection(c)) / len(c)
    return output


# In[131]:

def weightTuning(soNtags, tot_bic, tot_fic, k):
    final_eval = []
    eval_c = []
    for alpha in range(0, 10, 1):
        alpha = alpha / 10
        for beta in xrange(0, 10, 1):
            beta = beta / 10
#             print "Processing (%f, %f)" % (alpha, beta)
            eval_crit = 0
            for i in xrange(len(soNtags[0])):
                output = []
                
                bic = tot_bic_op[i]
                fic = tot_fic_op[i]
                
                tags = set(fic)
                tags.update(bic)
                
                for tag in tags:
                    value = alpha * bic.get(tag, 0) + beta * fic.get(tag, 0)
                    output.append((value, tag))
                    
                output.sort(reverse=True)
                
                actual_tags = soNtags[1][i]
                l = len(actual_tags)
                pred_tags = [o[1] for o in output[:l]]
                
                so = soNtags[0][i]
                
                evalVal = evalCrit(pred_tags, actual_tags)
                eval_c.append(evalVal)
                eval_crit +=  evalVal
            final_eval.append((evalVal, (alpha, beta)))
    final_eval.sort(reverse=True)
    np.array(eval_c)
#     print np.mean(eval_c)
    return np.mean(eval_c)


# In[132]:

avg_recall = weightTuning(soNtags, tot_bic_op, tot_fic_op, 1)


# In[133]:

def predictTags(alpha_beta, bic, fic, tags, k):
    alpha, beta = alpha_beta
    output = []
    for tag in tags:
        value = alpha * bic.get(tag, 0) + beta * fic.get(tag, 0)
        output.append((value, tag))
    output.sort(reverse=True)
    return output[:k]


# In[134]:

def getSOTags(so, correctTags, userid, user_history):
    test_probs = probability(so, tags, soNtags)
    test_bic = bic_output(diction, [so])[0]
    test_fic  = fic_output([test_probs], edge_wts)[0]
    
    try:
        user_index = user_history[0][user_history[0] == userid].index[0]
        user_tags = user_history[1][user_index].keys()
    except IndexError:
        user_tags = []
        
    predictedTags = predictTags((0.9, 0.9), test_bic, test_fic, tags, 5)
    
    ptags = []
    for p in predictedTags:
        ptags.append(p[1])
        
#     print "Predicted"
#     print predictedTags
    
    common_tags = list(set(correctTags).intersection(user_tags))
    
    final_tags = []
    
    if common_tags:
        for c in common_tags:
            if c not in predictedTags:
                final_tags.append(c)
    
    for p in predictedTags:
        if p[1] not in final_tags:
            final_tags.append(p[1])
    
#     return final_tags
    return ptags, final_tags


# In[137]:

# Pass preprocessed SO here
# getSOTags(pt[0], t[4][0], t[5][0], user_history)


# In[138]:

def get_accuracy(testfile, user_history):
    t = preprocessor(testfile)
    pt = pos_tag(t[3])
    best = 0
    avg = 0
    best_u = 0
    avg_u = 0
    for i in range(len(t)):
        param1 = pt[i]
        actual_tags = t[4][i]
        userid = t[5][i]
        final, final_user = getSOTags(param1,actual_tags, userid, user_history) 
#         print final
#         print actual_tags
        common = list(set(final).intersection(actual_tags))
        if common:
            best += 1
        avg += len(common)/len(actual_tags)
        
        common_u = list(set(final_user).intersection(actual_tags))
        if common_u:
            best_u += 1
        avg_u += len(common_u)/len(actual_tags)
    best_accu = best/len(t)
    avg_accu = avg/len(t)
    best_accu_u = best_u/len(t)
    avg_accu_u = avg_u/len(t)
    print "Best accuracy =     ",best_accu*100,"%"
    print "Average accuracy =  ",avg_accu*100,"%"
    print ""
    print "Best accuracy with UHIC =     ",best_accu_u*100,"%"
    print "Average accuracy with UHIC =  ",avg_accu_u*100,"%"
    return best_accu, avg_accu, best_accu_u, avg_accu_u


# In[140]:

user_history = uhic(pre[5],pre[4])
# best_accu, avg_accu, best_accu_u, avg_accu_u = getSOTags(pt[0],t[4][0], t[5][0], user_history) 


# In[142]:

g = get_accuracy('test.xml',user_history)

