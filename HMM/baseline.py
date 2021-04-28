# mp4.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created Fall 2018: Margaret Fleck, Renxuan Wang, Tiantian Fang, Edward Huang (adapted from a U. Penn assignment)
# Modified Spring 2020: Jialu Li, Guannan Guo, and Kiran Ramnath
# Modified Fall 2020: Amnon Attali, Jatin Arora
# Modified Spring 2021 by Kiran Ramnath
"""
Part 1: Simple baseline that only uses word statistics to predict tags
"""
from collections import Counter


def baseline(train, test):
    '''
    input:  training data (list of sentences, with tags on the tags)
            test data (list of sentences, no tags on the tags)
    output: list of sentences, each sentence is a list of (word,tag) pairs.
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    estimate = []
    wtag_dict = {}
    for sentence in train:
        for w_tag in sentence:
            if w_tag[0] not in wtag_dict:
                wtag_dict[w_tag[0]] = {w_tag[1]: 1}
            elif w_tag[1] not in wtag_dict[w_tag[0]]:
                wtag_dict[w_tag[0]][w_tag[1]] = 1
            else:
                wtag_dict[w_tag[0]][w_tag[1]] += 1

    reduce_ = {}
    cnt = Counter()
    for w, t in wtag_dict.items():
        high_p_tag = max(t, key = t.get)
        reduce_[w] = high_p_tag
        cnt += Counter(t)

    var = cnt.most_common(1)
    max_freq = var[0][0]

    for sentence in test:
        tagset = []
        for w in sentence:
            if w in reduce_:
                tagset = [*tagset, (w, reduce_[w])]
            elif w not in reduce_:
                tagset = [*tagset, (w, max_freq)]
        estimate = [*estimate, tagset]
    
    return estimate