#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  2 17:10:47 2017

@author: thanhan
"""

def extract_ne(sents):
    from nltk.tag import StanfordNERTagger
    import nltk
    
    st = StanfordNERTagger('ner/english.all.3class.distsim.crf.ser.gz', 'ner/stanford-ner.jar')
    
    
    
    sents_tk = []
    for sent in sents:
        sent_tk = nltk.word_tokenize(sent)
        sents_tk.append(sent_tk)
        
    
    ne = st.tag_sents(sents_tk)
    
    res = []    
    for sent in ne:
        last_tag = "O"
        en = ""
        sent.append(("", "O"))        
        
        for (word, tag) in sent:
            if tag == 'O':
                if en != "": res.append(en); en = ""            
            elif last_tag == tag:
                en += " " + word
            else:
                if en != "": res.append(en); en = ""
                en = word
            
            last_tag = tag
                
    return (ne, res)
        