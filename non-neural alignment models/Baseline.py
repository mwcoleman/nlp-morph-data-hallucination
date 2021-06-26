#!/usr/bin/env python
# coding: utf-8

# In[79]:


from collections import Counter
import numpy as np
import pandas as pd
import random


# In[50]:


## Takes two strings and computes the word alignment

def wordAlign(s1, s2):
    # Inputs: two strings - lemma and target inflected form
    # Outputs: (s1, s2) tuple of lists [pre, lemma, suff] 
    
    def substring(s):
        return [s[i: j] for i in range(len(s))
              for j in range(i + 1, len(s) + 1)]
    
    # s1 is smallest word
    rev=False
    if len(s2)<len(s1):
        rev=True
        s1, s2 = s2, s1
    
    # Return all substrings of s1 (descending length)
    # 
    subs = sorted(substring(s1), key = lambda x: len(x), reverse=True)
    
    # Find the largest alignment
    for sub in subs:
        if sub in s2:
            # Found longest match, split into pre stem suf..
            s2g = s2.split(sub, maxsplit=2) # doesn't include the sub
            s1g = s1.split(sub, maxsplit=2)
            stem = sub
            
            break
    
    # Reverse the reversal if needed
    if rev:
        s1g,s2g = s2g, s1g
    
    pre = (s1g[0], s2g[0])
    suf = (s1g[1], s2g[1])
    
    return (pre, stem, suf)
    

    

    


# In[51]:


## Step 2: Extract the rules for prefix and suffix changing

class Rule(object):
    ## Rule object for a given token feature set. 
    # Inputs: the Feature set token (string)
    #         and a list of all ((pre source, pre targ), stem, (suf source, suf target) ) for that token
    
    
    def __init__(self, token, tups): 
        # tups is a word-aligned list of nx3tuples, n = words in training set
        #token is the feature token string
        self.tups = tups
        self.token = token
        self.pre, self.stem, self.suf = zip(*self.tups)
        
        
        def recursiveSuf(self):
            # Extracts more suffix-generating rules by iterating through the stem
            
            rules = []
            for tup in tups:
                
                
                stem = tup[1]
                suf = tup[2][1]
                for i in range(len(stem), 0, -1):
                    rules.append((stem[i-1:], stem[i-1:]+suf))
            return rules
        
        # Add the extra suffix transformations 
        self.suf = list(self.suf) + recursiveSuf(self.tups)
        
        def setRules(tups):
            ruleDict = {}
            for i in range(len(tups)):
                s = tups[i][0]
                t = tups[i][1]

                if s in ruleDict:
                    ruleDict[s].append(t)
                else:
                    ruleDict[s] = [t]
            
            for k in ruleDict.keys():
                ruleDict[k] = dict(Counter(ruleDict[k]))
            return ruleDict
        self.pre = setRules(self.pre) 
        self.suf = setRules(self.suf)

    def token(self):
        return self.token

    def getRules(self):
        return (self.pre, self.suf)
    
    def applyRule(self, lemma):
        # takes a lemma and applies the inflection rule
        # suffix rule prioritises matching length, then frequency
        # prefix rule prioritises frequency
        
        inflected = lemma
        #First, suffix:
        
        for i in range(len(lemma)):
            if lemma[i:] in self.suf:
                # Found a match
                match = self.suf[lemma[i:]]
                # return the top match
                suf = list(sorted(match.items(), key = lambda x: x[1], reverse=True))[0][0]
                
                inflected = lemma[:i] + suf
#                 print(f"Found suffix match: {suf}, word becomes {inflected}. Now pre:")
        
        # now pre:
        if self.pre != {}:
            try:
                pre = list(sorted(self.pre[''].items(), key = lambda x: x[1], reverse=True))[0][0]
            except:
                pre = ''
        else:
            pre = ''
        return pre + inflected 


# In[52]:


def preCheck(df):
    # Checks if a language is primarily prefixing or suffixing 
    wa = []
    # Build up list of (pre, lemma, suff)
    for i in df.index:
        wa.append(wordAlign(df.loc[i,'lemma'], df.loc[i,'inflected']))
    # wa form : [ ((pre source, pre targ), stem, (suf source, suf target) ), .... ]#
    #Count prefixing
    count_pre = sum([p[0][1]!='' for p in wa])
    count_suf = sum([p[2][1]!='' for p in wa])
    return count_pre>count_suf

def buildModel(df, is_prefixing):
    rules = set(df['rule'])
    rule_set = {}
    
    
    # Reverse all words if language is primarily prefixing:
    if is_prefixing:
        for i in df.index:
            df.loc[i,'lemma'] = df.loc[i,'lemma'][::-1]
            df.loc[i,'inflected'] = df.loc[i,'inflected'][::-1]
    
    for r in rules:
        df_sub = df[df['rule']==r]
        wa = []
        for i in df_sub.index:
            wa.append(wordAlign(df_sub.loc[i,'lemma'], df_sub.loc[i,'inflected']))

        rule_set[r] = Rule(r, wa)
    
    return rule_set


def runModel(df, model, is_prefixing):
    # Takes a df of test data as per sigmorphon standard, and our model (a dictionary of rules)
    # returns a list of predictions
    
    # Reverse strings if is prefixing language
    if is_prefixing:
        for i in df.index:
            df.loc[i,'lemma'] = df.loc[i,'lemma'][::-1]
            
    
    pred = []
    for i in df.index:
        lemma = df.loc[i,'lemma']
        rule = df.loc[i,'rule']
        
        pred.append(model[rule].applyRule(lemma))
    
    if is_prefixing:
        pred = [w[::-1] for w in pred]
    return pred
    
    


# In[53]:


def getMetrics(pred, label):
    ## Compute accuracy and levenshtein distance given a list of predictions and a list of ground truths
    
    def levDistance(token1, token2):
        # https://blog.paperspace.com/implementing-levenshtein-distance-word-autocomplete-autocorrect/
        distances = np.zeros((len(token1) + 1, len(token2) + 1))

        for t1 in range(len(token1) + 1):
            distances[t1][0] = t1

        for t2 in range(len(token2) + 1):
            distances[0][t2] = t2

        a = 0
        b = 0
        c = 0

        for t1 in range(1, len(token1) + 1):
            for t2 in range(1, len(token2) + 1):
                if (token1[t1-1] == token2[t2-1]):
                    distances[t1][t2] = distances[t1 - 1][t2 - 1]
                else:
                    a = distances[t1][t2 - 1]
                    b = distances[t1 - 1][t2]
                    c = distances[t1 - 1][t2 - 1]

                    if (a <= b and a <= c):
                        distances[t1][t2] = a + 1
                    elif (b <= a and b <= c):
                        distances[t1][t2] = b + 1
                    else:
                        distances[t1][t2] = c + 1

        return distances[len(token1)][len(token2)]
    
    
    acc = 100*sum([pred[i]==label[i] for i in range(len(pred))])/len(pred)
    lev = sum([levDistance(pred[i], label[i]) for i in range(len(pred))])/len(pred)
    
    
    return (acc, lev)
    


# In[203]:


## RUN SCRIPT - Modify source data files as needed for different languages
if __name__ == "__main__":
	# # ENGLISH
	train = pd.read_csv("./data/english-train-low", sep='\t', names=["lemma", "inflected", "rule"])
	test = pd.read_csv("./data/english-dev", sep='\t', names=["lemma", "inflected", "rule"], index_col=False)
	test_label = list(test['inflected'])

	test_copy = test.copy()
	train_copy = train.copy()

	# If a test rule does not exist in the trainingf set, we find the nearest one and apply that
	test_rules = set(test['rule'])
	train_rules = set(train['rule'])
	random_rule = random.choice(train_copy['rule'])
	for tr in test_rules:
		if tr not in train_rules:
			for i in test_copy.index:
				if test_copy.loc[i,'rule']==tr:
					test_copy.loc[i,'rule']=random_rule

	test_label = list(test_copy['inflected'])

	is_prefixing = preCheck(train_copy)
	rs = buildModel(train_copy, True)

	# get predictions
	pred = runModel(test_copy, rs, True)

	acc, lev = getMetrics(pred, test_label)
	print(f"Accuracy: {round(acc,2)}% -- Average Levenshtein Distance: {round(lev,3)}")

