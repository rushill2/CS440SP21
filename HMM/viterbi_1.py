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
# Modified Spring 2021 by Kiran Ramnath (kiranr2@illinois.edu)

"""
Part 2: This is the simplest version of viterbi that doesn't do anything special for unseen words
but it should do better than the baseline at words with multiple tags (because now you're using context
to predict the tag).
"""
import numpy as np
import math
from sys import maxsize
from collections import Counter 

class tup:
    def __init__(self, p, tag):
        self.p = p                  # Probability
        self.tag = tag              # tag


# Helper function to compute the setupial train set transition and emission probabilities
# Parameters:
#           dict_ : input dict (transition or emission)
#           prev : the 'R0' stored in a variable
#           index1, index 2 : w, t 
#           flag : 0 for emission dict operations, 1 for transition
def probhelper(dict_, prev, index1, index2, flag):
    if flag ==0:
        if index1 not in dict_:
            dict_[index1] = {index2: 1}
        else:
            dict_[index1][index2] = dict_[index1].get(index2, 0) + 1
    elif flag ==1:
        if prev in dict_:
            if dict_[prev].get(index2) is not None:
                dict_[prev][index2] = dict_[prev].get(index2) + 1
                dict_[prev]['TOTAL'] = dict_[prev]['TOTAL'] + 1
            else:
                dict_[prev][index2] = 1
        else:
            dict_[prev] = {index2: 1, 
                'TOTAL': 1}

def listhelper(dict_, state, num, den):
    if den!=0:
        dict_[state] = math.log10(num/den)

def viterbi_1(train, test):
    estimate = []
    transition = {}
    endvar = 'END'
    sumvar = 'SUM'
    transition[endvar] = {}
    emission = {}
    emission[sumvar] = {}
    taglist = []
    k = 1e-12

    for sent in train:
        prev = 'R0'
        for (w, t) in sent:
            if t not in taglist:
                taglist = [*taglist, t]
            probhelper(transition, prev, w, t, 1)
            probhelper(emission, prev, w, t, 0)          
            prev = t
            if emission[sumvar].get(t) is not None:
                emission[sumvar][t] = emission[sumvar].get(t) + 1
            else:
                emission[sumvar][t] = 1

    trans_prob = {}
    tot = 'TOTAL'
    tcount = len(taglist) + 1
    for t in transition:
        dict_upd = {}
        for curr in transition[t]:
            if curr != tot:
                num_ = transition[t][curr] + k
                den_ = transition[t][tot] + k * tcount
                listhelper(dict_upd, curr, num_, den_)
        trans_prob[t] = dict_upd

    emission_pdict = {}
    wordlen = len(emission)
    wordlen += 1
    unseen = 'UNSEEN'
    emission_pdict[unseen] = {}
    for w in emission:
        dict_upd = {}
        if w != sumvar:
            for t in emission[w]:
                num = emission[w][t] + k
                den = emission[sumvar][t] + k * wordlen
                listhelper(dict_upd, t, num, den)
            emission_pdict[w] = dict_upd
    for t in emission[sumvar]:
        if emission[sumvar][t] is not None:
            den = emission[sumvar][t] + k * wordlen
            emission_pdict[unseen][t] = math.log10(k/den)
    str_ = ''
    for sent in test:
        setup = [[tup(0, str_) for s in sent] for l in taglist]
        tcount = len(taglist)
        i = 0
        for t in taglist:
            if t in trans_prob['R0']:
                transition_prob = trans_prob['R0'].get(t)
            else:
                if transition['R0'].get('TOTAL') is not None:
                    den = transition['R0'].get('TOTAL') + k * tcount
                    transition_prob = math.log(k / den)
            emission_prob = emission_pdict['UNSEEN'][t]

            if sent[0] in emission_pdict:
                if t in emission_pdict[sent[0]]:
                    emission_prob = emission_pdict[sent[0]].get(t)
            setup[i][0] = tup(emission_prob + transition_prob, 'R0')
            i += 1
        tot = 'TOTAL'
        for (i, w) in enumerate(sent):
            if i == 0:
                continue
            j = 0
            for curr in taglist:
                max_p = -maxsize
                ptr_prior = ''
                for prev, it in zip(taglist, range(len(taglist))):
                    prob_prev = setup[it][i - 1].p
                    if curr in trans_prob[prev]:
                        transition_prob = trans_prob[prev].get(curr)
                    else:
                        if transition[prev].get(tot) is not None:
                            den = transition[prev].get(tot) + k * tcount
                            transition_prob = math.log(k / den)
                    if max_p < transition_prob + prob_prev:
                        max_p = transition_prob + prob_prev
                        ptr_prior = prev
                
                if w in emission_pdict and curr in emission_pdict[w]:
                    if emission_pdict is not None:
                        emission_prob = emission_pdict[w][curr]
                else:
                    if emission_pdict['UNSEEN'][curr] is not None:
                        emission_prob = emission_pdict['UNSEEN'][curr]
                setup[j][i] = tup(max_p + emission_prob, ptr_prior)
                j += 1
        
        max_pprev = -maxsize
        ptr_prior = ''
        i = 0
        for t in taglist:
            prior = setup[i][-1]
            prob_prev = prior.p
            if t in trans_prob['END']:
                end = trans_prob['END']
                transition_prob = end.get(taglist)
            else:
                if transition[t].get('TOTAL') is not None:
                    den = transition[t].get('TOTAL') + k * tcount
                    transition_prob = math.log(k / den)
            state_prob = transition_prob + prob_prev
            if max_pprev < state_prob:
                max_pprev = state_prob
                ptr_prior = t
            i += 1

        state_t = ptr_prior
        rev = []
        tempset2 = []
        out = []
        i = len(sent)
        i -=1
        while state_t != 'R0':
            rev = [*rev, state_t]
            state_t = setup[taglist.index(state_t)][i].tag
            i -= 1
        tempset2 = rev[::-1]
        rev = tempset2
        for i, j in enumerate(sent):
            ans = (sent[i],rev[i])
            out = [*out, ans]
        estimate = [*estimate, out]
    return estimate