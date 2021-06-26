#!/usr/bin/env python
# coding: utf-8
# Author: Matthew Coleman for FIT5217 A2

from collections import Counter
import numpy as np
import pandas as pd
import re
import os
import sys
import random
import shutil
from bisect import bisect_left

def markov_hallucinate(s, t, markov_maps, augment_amount=10, stochastic=True,compositional=True, offset=0, ngram_size=1,lcs_min=0):
        
    def binarySearch(a, x):
        # Helper function to sample from ngram distribution based on cumsum
        i = bisect_left(a, x)
        if i:
            return (i-1)
        else:
            return -1
    
    def LCS(s1, s2):
        # Helper function: Find the longest common subsequence given two strings as input.
        # Outputs: (length of sequence), sequence
        # ref: https://stackoverflow.com/questions/48651891/longest-common-subsequence-in-python
        matrix = [["" for x in range(len(s2))] for x in range(len(s1))]
        for i in range(len(s1)):
            for j in range(len(s2)):
                if s1[i] == s2[j]:
                    if i == 0 or j == 0:
                        matrix[i][j] = s1[i]
                    else:
                        matrix[i][j] = matrix[i-1][j-1] + s1[i]
                else:
                    matrix[i][j] = max(matrix[i-1][j], matrix[i][j-1], key=len)

        cs = matrix[-1][-1]
        print(f"LCS length {len(cs)}  {cs}")
        return len(cs), cs

    s_copy = s[:]
    t_copy = t[:]
    
    # Extract lcs of the pair
    lcs_len, lcs = LCS(s,t)
    
    lcs_len_copy = lcs_len

    # store each consecutive subgroup of LSC along with its indices
    x = []
    i = 0
    while True:
        # The LCS contains all chars common to both in order, but not necessarily consecutive.
        # We break the LCS up into variables based on consecutive sequences common to both words.
        if i == len(lcs):
            break


        lcs_sub = lcs[:lcs_len-i]
        
        # Break loop if only e.g. single char matching left, and LCS was > 1 to begin with
        # if lcs_len_copy>1 and len(lcs_sub)<lcs_min:
        if len(lcs_sub)<lcs_min:
            break

        if (lcs_sub in s) and (lcs_sub in t):
            # Found common group: store the group, the index at s, the index at t
            x.append([lcs_sub, s.index(lcs_sub), t.index(lcs_sub)])
            
            # account for possible differences in position of the lcs_sub within each word
            # Shift to the next part of each word
            s = s[s.index(lcs_sub)+len(lcs_sub):]
            t = t[t.index(lcs_sub)+len(lcs_sub):]
            
            # Consider now only the remaining portion of the LCS and begin again
            lcs = lcs[lcs_len-i:]
            lcs_len = len(lcs)
            i = 0
         
        else:
            # No group found, reduce LCS subgroup size and try again
            i += 1
    if len(x)>1:
        print("More than one LCS consecutive subgroup found:")
        print(x)
        

    # Index of each variable currently is relative to last variable 
    # convert to absolute
    tallys = [a[1] for a in x]
    tallyt = [a[2] for a in x]
    for i in range(1,len(x)):
        # Each variable index gets shifted by the length of the 
        # variable value plus the previous variable position
        x[i][1] = x[i-1][1]+len(x[i-1][0]) + tallys[i]
        x[i][2] = x[i-1][2]+len(x[i-1][0]) + tallyt[i]
        
    def hallucinate(existed, markov_maps, source,target,lcs,mem='', bidirect=False):
        """Takes word pair in and returns another word pair with hallucinated chars replacing each lcs component
            
        Args:
            existed (list) : list of existing hallucanations of the word
            markov_maps ((d1,d2)) : d1 dict of chars and their [suf],[cumsum_freqs], d2 same for pre's. 
                                    Note: Only chars meeting length req are included
            source ([type]): source word
            target ([type]): target word
            lcs_inserts (list): list of lcs substrings, their insertion in s, their insertion at t
            bidirect : whether to consider backward pass or not (default not, recommended)
        """
        
        def pickSub(word,cumsum):
            # Helper func to pick a new substring from the distribution
            r = random.randint(1,cumsum[-1])
            sub = word[binarySearch(cumsum,r)+1]  
            return sub

        sw = list(r'#'+source+'$')
        tw = list(r'#'+target+'$')
        
        # Store substring and indices        
        ss = lcs[0]
        i_sw = lcs[1]
        i_tw = lcs[2]


        if (i_sw == len(sw)-1 or i_tw == len(tw)-1) and bidirect:
            # Then we replace with markov prefix choice
            # mm = markov_maps[0]
            mm = markov_maps[0]
        else:
            mm = markov_maps[1]
        
        # Store the possible words and their probabilities, 
        # based on the previous n-char (markov assumption, default 1)
        # subwords seen twice sum their likelihood

        word = mm[sw[i_sw]][0] + mm[tw[i_tw]][0]
        cumsum = mm[sw[i_sw]][1] + mm[tw[i_tw]][1]
        
        # Adjust cumsum due to merging 
        for i in range(len(mm[tw[i_tw]][1])):
            cumsum[len(mm[sw[i_sw]][1]) + i] = cumsum[len(mm[sw[i_sw]][1]) + i] + cumsum[len(mm[sw[i_sw]][1]) + i - 1]

        # Remove whitespace and any already used
        nw = []
        nc = []
        for i in range(len(word)):
            if word[i] == '' or word[i] in existed:
                continue
            else:
                nw.append(word[i])
                nc.append(cumsum[i])
        word, cumsum = nw,nc    
        # print(f"Chars succeeding {sw[i_sw]} in source word or {tw[i_tw]} in target word:" )
        # print(list(zip(word,cumsum)))
        # print('------')
        if compositional==False:
            # If not allowing compositional elements, then we filter words to within offset and return 
            word,cumsum = zip(*[(w,c) for w,c in zip(word,cumsum) if len(w)>=len(ss)-offset and len(w)<=len(ss)+offset])

        if word == []:
            # No other option.. return as is
            return sw,tw,existed

        # Choose substring from options:
        sub = pickSub(word,cumsum)
        sub = sub[:min(len(sub),len(ss))]

        mem += sub
        
        # Store old copies, in case of repeated replacement, we call recursive with these instead
        sw_old = ''.join(sw[:]).strip('#$')
        tw_old = ''.join(tw[:]).strip('#$')

        sw = ''.join(sw[0:i_sw+1] + list(sub) + sw[i_sw+len(ss):]).strip('#$')
        tw = ''.join(tw[0:i_tw+1] + list(sub) + tw[i_tw+len(ss):]).strip('#$')
        
        
        if not compositional:
            # Then the chosen word has to be within the offset limit we chose, and simply return it
                existed.append(mem)
                return sw,tw,existed
        
        elif compositional:
            if (len(sub) == len(ss)):
                # We have a winner
                if mem in existed:
                    # print(f"\n\n Replacement was: {mem} but that is in existed")
                    return hallucinate(existed,markov_maps,sw_old,tw_old,lcs,mem)
                else:
                    existed.append(mem)
                    # print(f"Found replacmeent: {mem+sub}. Words are: {sw}  - {tw} --  Existed becomes: \n{existed}")
                    return sw,tw,existed
            
            else:
                # Recursive call with the remaining portion of the lcs as the new lcs
                new_lcs = [lcs[0][len(sub):], lcs[1]+len(sub), lcs[2]+len(sub)]
                # print(f"replacement is not equal length, recursively calling with params: \nexisted:{existed},source:{sw},target:{tw}, \nmem:{mem}\nlcs:{new_lcs}")
                return hallucinate(existed,markov_maps,sw,tw,new_lcs,mem+sub)

    max_iter = 200
    j = 0
    i = 0
    new_sources = []
    new_targets = []

    for lcs_sub in x:
        existed = []
        existed.append(lcs_sub[0])    
        while i < augment_amount:
            j += 1 # Stop if max iter
            if j > max_iter:
                print("Breaking due to max iterations")
                break
            
            sw,tw,existed = hallucinate(existed,markov_maps,s_copy,t_copy,lcs_sub)
            
            # Still seem to be creating duplicates, fix this in future. For now, just skip
            if sw not in new_sources:
                i += 1
                new_sources.append(sw)
                new_targets.append(tw)
        # print(f"Old words: {s_copy}, {t_copy}\n New words: {new_sources}, \n{new_targets}")
    return new_sources, new_targets


def markov(wordlist, max_len = 4, ss_len = 1):
    # Returns 2 dict mapping substrings (ss) of length ss_len to their observed prefix and suffixes
    
    mm_suf = dict()
    mm_pre = dict()
    ss_list = []
    

    def cumSum(mm):
        """Returns a cumulative sum of word frequencies in form of [[words],[cumsum]]
        Args:
            mm (dict): dict of count objects of word frequencies
        """
        counts = dict()
        cumsum = dict()

        # Refactor so each key has a affixes and their counts seperated
        counts = {k:list(zip(*list(mm[k].items()))) for k in mm.keys()}
        
        # Calcualte cumsum
        for k in counts:
            c = 0
            cumsum[k] = [list(counts[k][0]),[]]
            for i in range(len(counts[k][1])):
                c += counts[k][1][i]
                cumsum[k][1].append(c)
        return cumsum

    # prepend with start and end of word - TODO: maybe change when we do this... depends if want in the count dicts.. or just as keys in the markov..
    wordlist = [r'#'+w+r'$' for w in wordlist]
    # Get list of all substrings of length prefix_len
    for w in wordlist:
        ss_list += [w[i: i+ss_len] for i in range(len(w)-ss_len)]
    # Store set of all substrings, or all chars (in the case of ss_len=1 default)
    ss_list = set(ss_list) 
    
    # Case when we want unigram dict, iterate through each char.
    if ss_len==1:
        
        for ss in ss_list:
            for w in wordlist:
                ssi = 0
                for c in w:
                    if ss == c:
                        if ssi!=(len(w)-len(ss)):
                            suf = w[ssi+len(ss): ssi+len(ss)+min(max_len, len(w))].replace('#','').replace('$','')
                            mm_suf[ss] = mm_suf.get(ss, []) + [suf]
                        if ssi!=0:
                            pre = w[ssi-min(ssi,max_len):ssi].replace('#','').replace('$','')
                            mm_pre[ss] = mm_pre.get(ss, []) + [pre]
                    ssi += 1

    # case when we using ngram dict, just match greedy on the substring
    else:

        for ss in ss_list:
            for w in wordlist:
                # If the prefix is in the word, and it isnt the end of the word
                if (ss in w):
                    ssi = w.find(ss)
                    if ssi!=(len(w)-len(ss)):
                        suf = w[ssi+len(ss): ssi+len(ss)+min(max_len, len(w))].replace('#','').replace('$','')
                        mm_suf[ss] = mm_suf.get(ss, []) + [suf]
                    if ssi!=0:
                        pre = w[ssi-min(ssi,max_len):ssi].replace('#','').replace('$','')
                        mm_pre[ss] = mm_pre.get(ss, []) + [pre]
    
    
    
    mm_suf = {k:Counter(v) for k,v in mm_suf.items()}
    mm_pre = {k:Counter(v) for k,v in mm_pre.items()}
    # print(mm_pre)
    return cumSum(mm_pre), cumSum(mm_suf)



def run(lang, version, offset=0,ngram_size=1, stochastic=True,compositional=True, data_size=5000, lcs_min=0, max_len=4):
    """[summary]
    writes output csv to augmented folder and copies to each models data dir

    Args:
        lang (str): the name of the base language
        version (str): version of augment- used for labelling
        offset (int, optional): Should the replacement match the size of the original substring?. Default = 0 is matched.
        ngram_size (int, optional): ngram size i.e. how far back to consider. Defaults to 1. NOT IMPLEMENTED
        stochastic (bool, optional): if stochastic=True (def) sample from observered ngram distribution.  NOT IMPLEMENTED
                                     if false, creates every observed combination (overriding dataset size, compositional,block_size). Defaults to True .
        compositional (bool, optional): make up substitutions recursively e.g. replace a substring of length 3 with 2+1 ngram. Defaults to True.
        target_datasize (int, optional): Desired amount to augment dataset. Not strictly enforced. Defaults to 5000.
        max_len (int, optional): the maximum char length to save for the vocabulary. eg. if = 1, then ngram vocab contains only letters w/ their observed freq.
    """

    # Get data
    
    # You may need to change the current working directory 
    #os.chdir('/home/matt/fit5217/mc_augment/')
    
    file_loc = "./"+lang+".trn"
    try:
        train = pd.read_csv(file_loc, sep='\t', names=["lemma", "inflected", "rule"])
    except:
        print("Make sure the current working directory is set to the scipt path, \n \
            and the .trn/.tst/.dev files are there")
    train_len = train.shape[0]

    pairs = list(zip(train['lemma'], train['inflected']))
    rules = [train['inflected'][i] for i in range(len(pairs))]
    
    # Each word generates h new words
    h = (data_size-train_len)/train_len

    # concat lemma and inflected to extract subwords    
    all_words = train['lemma'].append(train['inflected'])

    hall_df = pd.DataFrame(None,columns=['lemma','inflected','rule'])
    rule_index = 0
    for s,t in pairs:

        hall_source, hall_target = markov_hallucinate(s, t, markov(all_words, max_len=max_len), augment_amount=h, lcs_min=lcs_min)
        
        # Add on the original (real) words
        hall_source.append(s)
        hall_target.append(t)

        df = pd.DataFrame(zip(hall_source, hall_target, [train['rule'][rule_index] for i in range(len(hall_source))]), columns=['lemma','inflected','rule'])
        hall_df = pd.concat([hall_df,df])
        rule_index += 1
    
    out_name = lang+'_'+version
    # Make copies of the dev and test sets put them in a augment folder and copy folder over to each model dir
    if out_name not in os.listdir(os.getcwd()):
        os.mkdir(out_name)

    path_to_out = './'+out_name+'/'
    hall_df.to_csv(path_to_out+out_name+'.trn', sep='\t', index=False, header=False)
    
    
    for src, dst in zip([lang+'.dev', lang+'.tst'], [out_name+'.dev', out_name+'.tst']):
        print(f"Copying {src} to {path_to_out+dst}")
        shutil.copy(src, path_to_out+dst)

## Uncomment if you want to do a batch augmentation, adjust params etc.:

# for l in ['mwf','izh','mlt']:
#     for o in [0,1]:
#         for c in ['c','f']:
#             for lcs_m in [2,3]:
#                 print(f"Running for {l}_d2k_o{o}_{c}_lcs-{lcs_m}")
#                 run(l,f'd2k_o{o}_{c}_lcs-{lcs_m}', offset=o, compositional=(c=='c'), data_size=2000,lcs_min=lcs_m, max_len=10)



 # hopefully this works.. otherwise just remove it, uncomment the above, and run the script.
if __name__ == "__main__":
    try:
        lang = sys.argv[1]
        vers = sys.argv[2]
        ds = sys.argv[3]
        if len(sys.argv) > 3:
            offset = sys.argv[4]
            comp = sys.argv[5]
            lcs_m = sys.argv[6]
            max_len = sys.argv[7]
        else:
            offset = 0
            comp = False
            lcs_m = 3
            max_len = 10

        run(lang,verse, offset=offset, compositional=comp, 
                                    data_size=ds,lcs_min=lcs_m, max_len=max_len)
    except:
        print("Error passing arguments. \n call: \
        'python markov_hall.py [lang_name] [augment_name] [augmented_size]' \
            at a minimum")
        print("Where [lang_name] is the filename of the train/test/dev files \
            excl. extension. must have .trn/.dev/.tst extensions)\n\n and \
                [augment_name] is version to save augmented data to"
         