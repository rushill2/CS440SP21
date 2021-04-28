# naive_bayes.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 09/28/2018

"""
This is the main entry point for Part 1 of this MP. You should only modify code
within this file for Part 1 -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""
# ham_dict = {}
# spam_dict = {}
# tag = 0


parent_dict = {
        'tag': 0,
        'dict1': {},
        'dict2': {}
    }

import numpy as numpy
import math
from collections import Counter
from collections import defaultdict

def slicing_helper(var, dict_, slice_):
    if slice_ not in dict_:
        dict_[slice_] = 1
    else:
        dict_[slice_] += 1

def switch_helper(slice_):
    dict1 = parent_dict.pop('dict1')
    dict2 = parent_dict.pop('dict2')
    tag = parent_dict.pop('tag')
    if(tag==1):
        slicing_helper(tag, dict1, slice_)
    else:
        slicing_helper(tag, dict2, slice_)

def naiveBayes(train_set, train_tags, dev_set, smoothing_parameter, pos_prior):

    # TODO: Write your code here		    
    prob = 0
    i=0
    j=0
    k=0
    ham_dict = {}					    
    spam_dict = {}
    log_priorprob = math.log10(pos_prior)
    log_negprior = math.log10(1-pos_prior)
    twice_smooth = smoothing_parameter*2
    # print(len(train_set))
    while i<len(train_set):     
        rev_var = train_set[i]
        tag = train_tags[i]
        for j in range(len(rev_var)):    
            slice_ = rev_var[j]
            parent_dict['dict1'] = ham_dict
            parent_dict['dict2'] = spam_dict
            parent_dict['tag'] = tag
            switch_helper(slice_)
        i+=1            
    spam_filter = []
    dict1_vals = ham_dict.values()
    dict2_vals = spam_dict.values()							
    # print(len(dev_set))
    while k<len(dev_set):           		
        content = []
        for x in dev_set[k]:
            # print(word)
            content = [*content, x]
        # print(content) 
        prob = 0         
        indices = sum(dict1_vals)+twice_smooth
        for word in content:
            if word not in ham_dict:
                ham_dict.setdefault(word, 0)	
            prob += math.log10((ham_dict[word]+smoothing_parameter)/(indices))
            
        prob_safe = prob+log_negprior
        probneg = 0
        # for x in spam_dict.values():
        #     indices+=x
        indices = sum(dict2_vals)+twice_smooth
        # print(type(ham_dict.values()))
        for word in content:
            if word not in spam_dict:
                spam_dict.setdefault(word, 0)	
            probneg += math.log10((spam_dict[word]+smoothing_parameter)/(indices))
            
        prob_safe = prob+log_priorprob
        prob_spam = probneg+math.log10(1-pos_prior)
        # lenvar = len(spam_filter)
        spam_filter=[*spam_filter, 0] if (prob_safe < prob_spam) else [*spam_filter, 1]
        k+=1
    return spam_filter


   