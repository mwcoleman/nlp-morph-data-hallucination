#!/usr/bin/env python
# coding: utf-8

## Run script is at the bottom - modify as needed for different languages

# In[355]:


from collections import Counter
import numpy as np
import pandas as pd
import re
from sklearn.linear_model import LogisticRegression



def generaliser(s, t):
    # Inputs: two forms of a word, s: source form, t: target (inflected) form
    # Outputs: an abstract paradigm representation of the transform with integers representing variables
    #          e.g. swim , swam => 0i1#0a1 
    #          function returns both string representation and list element representation
    #          e.g. ('0i1#0a1', [[0,'i',1],[0,'a',1]])
    # Abstract paradigms reference: "Semi-supervised learning of morphological paradigms and lexicons" 
    #                                - Ahlberg et al 2014


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

        return len(cs), cs


    # Create backup copy for later. we modify s & t below. 
    # s: source form, t: target form
    s_copy = s[:]
    t_copy = t[:]
    
    # Extract lcs of the pair
    lcs_len, lcs = LCS(s,t)
    
    # for each common segment/grouping of the LCS
    # we will store it as a variable
    x = []
    i = 0
    while True:
        # The LCS contains all chars common to both in order, but not necessarily consecutive.
        # We break the LCS up into variables based on consecutive sequences common to both words.
        if i == len(lcs):
            break      
        
        lcs_sub = lcs[:lcs_len-i]
        
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
    

    # Index of each variable currently is relative to last variable, 
    # convert to absolute
    tallys = [a[1] for a in x]
    tallyt = [a[2] for a in x]
    for i in range(1,len(x)):
        # Each variable index gets shifted by the length of the 
        # variable value plus the previous variable position
        x[i][1] = x[i-1][1]+len(x[i-1][0]) + tallys[i]
        x[i][2] = x[i-1][2]+len(x[i-1][0]) + tallyt[i]
    
    def getRep(word, x):
        # Helper function to get the abstract paradigm representation
        # Inputs: a word and a list of found variables with their insertion position
        # Outputs: abstract paradigm as string and list form.
        sub = ''
        char = ''
        sublist = []
        has_been_char = False
        i=0
        j=0
        while i < len(word):
            # Proceed through word until insertion index of variable, 
            # substituting them when reached
            if i<x[j][1]:
                sub += word[i]
                char += word[i]
                i += 1
            else: 
                if char != '':
                    sublist.append(char)
                    char = ''
                    
                sublist.append(j)
                sub += f"{j}"      
                i += len(x[j][0])
                j += 1
                if j == len(x):
                    sub += word[i:]
                    sublist.append(word[i:])
                    break
        return (sub, sublist)
  
    # Refactor the x variable list into source and target specific tuples
    x_s = [(x[i][0],x[i][1]) for i in range(len(x))]
    x_t = [(x[i][0],x[i][2]) for i in range(len(x))]
    
    # Store the abstract paradigm for each part
    s_rep = getRep(s_copy, x_s)
    t_rep = getRep(t_copy, x_t)

    return (s_rep[0]+'#'+t_rep[0], (s_rep[1],t_rep[1]))
            


def ftExt(word):
    # Simple function to extract prefixes and suffixes for use in logistic classifier
    # given a word it returns list of all subsets of prefixes <4 in length, and suffixes <6 in length.
    pre = []
    suf = []
    for i in range(min(len(word),5)):
        suf.append(word[len(word)-i:]+'$')
        if i<3:
            pre.append('^'+word[:3-i])
    return pre+ suf[1:]            
    
    # ftExt("sang")
    # => ['^san', '^sa', '^s', 'g$', 'ng$', 'ang$']


# In[684]:


## Class rules
class Rule(object):
    ## Rule object for a given token feature set. 
    # Inputs: the Feature set token representing tags(string), train dataset, test dataset
    # Uses a predictive model to estimate a word's inflected type from a lemma
    # using a logistic classifier on the training set with labels = abstract paradigm representations
    
    
    def __init__(self, token, train, test): 
        #token is the feature token string
        self.train = train[train['rule']==token]
        self.test = test[test['rule']==token]
        self.token = token
        
        
    def genParadigmTable(self):
        pairs = list(zip(self.train['lemma'], self.train['inflected']))
        paradigms = [generaliser(p[0],p[1])[0] for p in pairs]
        rules = [self.token for i in paradigms]
        all_together = list(zip(self.train['lemma'], self.train['inflected'], paradigms, rules))
        pTable = pd.DataFrame(all_together, columns=['source', 'target', 'paradigm', 'rule'])
        return pTable
    
    def buildData(self):
        # Converts train and test df inputs into a set with all pre/suff-ix features from the training set
        
        # 1) Create a training dataframe by extracting pre/suf-ix features and labels (abstract paradigms) 
        all_ft = [ftExt(word) for word in self.train['lemma']]
        all_ft_flat = [item for sublist in all_ft for item in sublist]
        pairs = list(zip(self.train['lemma'], self.train['inflected']))

        paradigms = [generaliser(p[0],p[1]) for p in pairs]
        
        # Store the AP string to list kv pairs, for instantiating later on.
        self.paradigm_dict = {p[0]:p[1] for p in paradigms}
        self.paradigm_dict['0#0'] = [[0,''],[0,'']]
        # For this part deal only with the string representation
        paradigms = [p[0] for p in paradigms]
        
        counts = Counter(all_ft_flat)
        counts_copy = counts.copy()
            
        self.training_features = counts.keys()
        
        # Create dataframe
        train_df = pd.DataFrame(columns=self.training_features)
        train_df['lemma']=self.train['lemma']
        train_df['paradigm'] = paradigms

        i = 0
        for lemma in train_df['lemma']:
            for col in train_df.columns:
                if col in all_ft[i]:
                    train_df.loc[train_df['lemma']==lemma,col] = 1
            i += 1
        train_df.fillna(0, inplace=True)
        
        self.train_onehot = train_df
    
        ## Now do the same for the test set, except we use the features extracted form the training set 
        # as our dataframe columns
        all_ft = [ftExt(word) for word in self.test['lemma']]
        test_df = pd.DataFrame(columns=self.training_features)
        test_df['lemma'] = self.test['lemma']
        test_df['inflected_truth'] = self.test['inflected']
        i = 0
        for lemma in test_df['lemma']:
            for col in test_df.columns:
                if col in all_ft[i]:
                    test_df.loc[test_df['lemma']==lemma, col] == 1
        test_df.fillna(0, inplace=True)
        
        self.test_onehot = test_df
    
    def trainModel(self):
        X = self.train_onehot[:]
        X = X.drop(['lemma', 'paradigm'], axis = 1)
        y = self.train_onehot['paradigm']
        softmax_reg = LogisticRegression(multi_class="multinomial",solver="lbfgs", C=10)
        softmax_reg.fit(X,y)
        
        self.model = softmax_reg
    
    def predict(self):
        # Takes a test lemma and
        # returns a predicted abstract class
        try:
            itruth = self.test_onehot['inflected_truth']
            df = self.test_onehot.drop(['lemma', 'inflected_truth'], axis=1)
        except:
            df = self.test_onehot.drop(['lemma'], axis=1)
        
        # If no features from this word are found in the training set, then AP-> 0#0
        if df.shape[0]==0:
            df['lemma'] = self.test_onehot['lemma']
            df['pred_paradigm'] = ['0#0' for i in self.test_onehot]
        else:
            df['pred_paradigm']=self.model.predict(df)        
            df['lemma'] = self.test_onehot['lemma']
            self.predicted_ap = df[['lemma', 'pred_paradigm']]
    
    def getExport(self):
        export = self.predicted_ap.copy()
        other = self.test_onehot.copy()
        export['Truth'] = other['inflected_truth']
        export['rule'] = [self.token for i in range(export.shape[0])]
        return export
    
    def getAP(self):
        return self.predicted_ap
    
    def setAP(self, label):
        # Set the paradigm in the case there is only one (i.e. no classificaiton problem)
        df = self.test_onehot
        df = df.assign(pred_paradigm=[label for i in range(df.shape[0])])
        self.predicted_ap = df[['lemma', 'pred_paradigm']]
        
    def guessLemma(self):
        for i in self.predicted_ap.index:
            print(self.predicted_ap.iloc[i,])
    
    def getToken(self):
        return self.token
    
    def getTrainData(self):
        return self.train_onehot
    
    def instantiate(self):
#         takes the predicted abstraction data and the lemma, and attempts to match variables to sections
#         returning a predicted target word form
        df = self.predicted_ap
        
        for i in df.index:
#             print(f"Considering df row:\n {df.loc[i,]}")
            lemma = df.loc[i,'lemma']
            
            ap_label = df.loc[i,'pred_paradigm']
            ap = self.paradigm_dict[ap_label]
            
            r = r''
            ap_new = []
            last = ''
            for e in ap[0]:
                if e=="":
                    continue
                elif type(e)==int:
                    ap_new.append(e)
                    r = r+r'(\D+)'
                else:
                    ap_new.append(e)
                    r = r+r'('+e+r')'
               # Wrap regex matching in try backets for 
               # case of incorrect AP patterns            
            try:
                reg = re.compile(r)
                mo = reg.search(lemma)
                mo = mo.groups()

                # Now assign variables:

                var_assign = dict()
                for e in ap[0]:
                    if type(e)==int:
                        var_assign[e] = mo[ap_new.index(e)]

                # Now form the prediction
                inf = ''
                for e in ap[1]:
                    if type(e)==int:
                        inf += var_assign[e]
                    else:
                        inf += e

                df.loc[i,'pred_inflected'] = inf
            
            except:
                dumb_suffix = ap_label[ap_label.index('#')+1:]
                d = ''
                for c in range(len(dumb_suffix)):
                    try:
                        int(dumb_suffix[c])
                    except:
                        d += dumb_suffix[c]
                df.loc[i,'pred_inflected'] = df.loc[i,'lemma']+d
        self.predicted_ap = df
            
            

    


# In[711]:


def run(train, test):
    ## Preproces data so the test 

    # The prediction process runs for each token as a subset
    # We generate a new rule object for each token, which controls the whole process
    ruleset = []    
    t = 0
    train_rules = set(train['rule'])
    test_rules = set(test['rule'])
    
    rules = [r for r in test_rules if r in train_rules]
    
    for rule in rules:
        nr = Rule(rule, train, test)
        nr.buildData()

        # Check if there are multiple classes-> if not, no point training..
        if len(set(nr.getTrainData()['paradigm'])) > 1:
            nr.trainModel()
            nr.predict()
        else:
            nr.setAP(list(nr.getTrainData()['paradigm'])[0])
        
        # Apply the predicted paradigms to the test lemmas
        nr.instantiate()
        # Add     
        ruleset.append(nr)
    
    # Now refactor the output dataset
    df = ruleset[0].getAP()
    for rule in ruleset[1:]:
        df = pd.concat((df,rule.getAP()))
    
    # Deal with any unseen rules - repeat lemma:
    test_rules = set(test['rule'])
    for rule in test_rules:
        if rule not in rules:
            tdf = test[test['rule']==rule]
            for i in tdf.index:
                df.loc[i,'lemma']=tdf.loc[i,'lemma']
                df.loc[i,'pred_paradigm']=tdf.loc[i,'lemma']
                df.loc[i,'pred_inflected']=tdf.loc[i,'lemma']
    df = df.sort_index()
    return df, ruleset


# In[ ]:


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


# In[729]:


## RUN Script - Modify as needed for different language sets
if __name__ == "__main__":
	# Get data
	train = pd.read_csv("./data/turkish-train-medium", sep='\t', names=["lemma", "inflected", "rule"])
	test = pd.read_csv("./data/turkish-uncovered-test", sep='\t', names=["lemma", "inflected", "rule"], index_col=False)
	test_label = list(test['inflected'])

	df, ruleset= run(train, test)

	acc, lev = getMetrics(df['pred_inflected'], test_label)
	print(f"Accuracy: {round(acc,2)}% -- Average Levenshtein Distance: {round(lev,3)}")

	# output = test.copy()
	# output['Prediction'] = df['pred_inflected']
	#output.sort_values(by='rule').to_csv('AP_predictions.csv', sep=',') 

