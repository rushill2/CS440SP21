# naive_bayes_mixture.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 09/28/2018
# Modified by Jaewook Yeom 02/02/2020

"""
This is the main entry point for Part 2 of this MP. You should only modify code
within this file for Part 2 -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""



import numpy as numpy
import math
from collections import Counter

parent_dict = {
        'tag': 0,
        'dict1': {},
        'dict2': {}
    }

def slicing_helper(dict_, slice_):
    if slice_ not in dict_:
        dict_[slice_] = 1
    else:
        dict_[slice_] += 1

def switch_helper(slice_):
    dict1 = parent_dict['dict1']
    dict2 = parent_dict['dict2']
    tag = parent_dict['tag']
    if(tag == 1):
        slicing_helper(dict1, slice_)
    else:
        slicing_helper(dict2, slice_)

def naiveBayesMixture(train_set, train_tags, dev_set, bigram_lambda,unigram_smoothing_parameter, bigram_smoothing_parameter, pos_prior):
    
     # TODO: Write your code here
    # Edge case 
    if(bigram_lambda<0 or bigram_lambda==0):
        print("MathError caused ")
        return
    # setting up iterator variables
    i = 0
    j = 0
    k = 0

    # Dicts for ham and spam 
    bigram_ham = {}                   
    bigram_spam = {}    

    # parameters for neg and pos big and uni              
    twice_smooth_bi = 2*bigram_smoothing_parameter
    twice_smooth_uni = 2*unigram_smoothing_parameter
    prob_prior = math.log10(pos_prior)
    inv_prob_prior = math.log10(1-pos_prior)

    # iterating thru the train_set
    while i<(len(train_set)):    
        rev_set = train_set[i]
        tag = train_tags[i]

        for j in range(len(rev_set)-1):      
            word1 = rev_set[j]
            word2 = rev_set[j+1]
            tup = (str(word1), str(word2))
            bigram = ''.join(tup)

            # Pushing to global dict for usage elsewhere
            parent_dict['dict1'] = bigram_ham
            parent_dict['dict2'] = bigram_spam
            parent_dict['tag'] = tag
            switch_helper(bigram)
        i+=1


    spam_filter = []                        
    ham_dict_uni = {}                   
    spam_dict_uni = {}                   
    i=0

    # Getting the words 
    while i<len(train_set):     
        rev_set = train_set[i]
        tag = train_tags[i]

        for j in range(len(rev_set)):    #
            slice_ = rev_set[j]
            parent_dict['dict1'] = bigram_ham
            parent_dict['dict2'] = bigram_spam
            parent_dict['tag'] = tag
            switch_helper(slice_)
        i+=1
    while k<(len(dev_set)):       
        dev_unit = dev_set[k]
        content_bigram = []              
        for l in range(len(dev_unit)-1):       
            tup = (str(dev_unit[l]), str(dev_unit[l+1]))
            bigram = ''.join(tup)
            content_bigram = [*content_bigram,  bigram]
        k+=1

        # Ham prob for bigram 
        prob = 0
        dict1_vals = bigram_ham.values()         
        indices = sum(dict1_vals)+twice_smooth_bi
        for word in content_bigram:
            if word not in bigram_ham:
                bigram_ham.setdefault(word, 0)	
            prob += math.log10((bigram_ham[word]+bigram_smoothing_parameter)/(indices))
        pos_bigram = prob + prob_prior

        # Spam prob for bigram 
        probneg = 0
        dict2_vals = bigram_spam.values()         
        indices = sum(dict2_vals)+twice_smooth_bi
        for word in content_bigram:
            if word not in bigram_spam:
                bigram_spam.setdefault(word, 0)	
            probneg += math.log10((bigram_spam[word]+bigram_smoothing_parameter)/(indices))
        neg_bigram = probneg+ inv_prob_prior

        #Setting up word content dev_unit for unigram
        content_unigram = []             
        for word in dev_unit:
            content_unigram = [*content_unigram, (word)]

        # Ham prob for unigram
        prob = 0   
        float(prob)      
        indices = sum(ham_dict_uni.values())
        for word in content_unigram:
            if word not in ham_dict_uni:
                ham_dict_uni.setdefault(word, 0)         
            prob += math.log10((ham_dict_uni[word]+unigram_smoothing_parameter)/(indices+(twice_smooth_uni))) # 2 since types are ham_dict & spam_dict
        
        pos_unigram = prob+ prob_prior

        # Spam prob for unigram 
        prob = 0   
        float(prob)      
        indices = sum(spam_dict_uni.values())
        for word in content_unigram:
            if word not in spam_dict_uni:
                spam_dict_uni.setdefault(word, 0)         
            prob += math.log10((spam_dict_uni[word]+unigram_smoothing_parameter)/(indices+(twice_smooth_uni))) # 2 since types are ham_dict & spam_dict
        
        neg_unigram = prob+ inv_prob_prior

        # Loop edge case
        if (1-bigram_lambda) <= 0:
            spam_filter = [*spam_filter, -1]              # mathError flag
            continue

        # Getting total prob
        prob_total_spam = 0
        prob_inv_bigram = math.log10(1-bigram_lambda)
        prob_bigram = math.log10(bigram_lambda)
        prob_total_ham = prob_inv_bigram +pos_unigram
        prob_total_spam += prob_bigram
        prob_total_spam = prob_inv_bigram +neg_unigram
        prob_total_spam += prob_bigram
        ham_dict_prob = prob_total_ham + pos_bigram
        spam_dict_prob = neg_bigram + prob_total_spam

        # Pushing results to returned dict
        spam_filter=[*spam_filter, 0] if (ham_dict_prob < spam_dict_prob) else [*spam_filter, 1]

    return spam_filter