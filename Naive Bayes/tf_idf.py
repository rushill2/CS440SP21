# tf_idf_bayes.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 09/28/2018
# Modified by Jaewook Yeom 02/02/2020
# Modified by Kiran Ramnath 02/13/2021

"""
This is the main entry point for the Extra Credit Part of this MP. You should only modify code
within this file for the Extra Credit Part -- the unrevised staff files will be used when your
code is evaluated, so be careful to not modify anything else.
"""

import numpy as np
import math
from collections import Counter, defaultdict
import time
import operator

docLength = {}


def tf_idf(train_set, train_labels, dev_set):
    # TODO: Write your code here
    
    while i <len(train_set):		
        review = train_set[i]
        label = train_labels[i]
        for word in review:				
            if word not in docLength:
                docLength[word] = 1
            else:
                docLength[word] += 1
        i+=1

    wordset = []

    for i in range(len(dev_set)):		
        review = dev_set[i]
        wordList = [] 					
        wordCount = {} 					
        for word in review:				
            if word not in wordCount:
                wordCount[word] = 1
            else:
                wordCount[word] += 1
            wordList.append(word)
        total = len(wordList)
        tf = {}
        for word in review:				
            if word not in tf:
            	if (word not in docLength):
            		docLength[word] = 0
            	tf[word] = (wordCount[word]/total) * math.log10((len(train_set))/(1+docLength[word]))

        biggy = -99.999					
        warad = ""
        for key, value in tf.items():	
            if value > biggy:			
                biggy = value
                warad = key
        wordset = [*wordset, warad]

    # return list of words (should return a list, not numpy array or similar)
    return wordset