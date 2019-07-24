# -*- coding: utf-8 -*-
"""
Created on Sun Aug 07 17:32:48 2016

@author: Rahul Swami
"""
from __future__ import division
from __future__ import print_function
import sys
import numpy as np
import pandas as pd
import re
import nltk
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer

def keep_it_strict(token):
    """ This function returns whether a unigram is part of the stop words and special characters or not.
    If it is not part of stop words,enpuncts and others,Then return false else true.The stop words list 
    is imported from nltk.corpus.It is used for cleaning the summary tokens"""
    from nltk.corpus import stopwords
    enStops = stopwords.words('english')
    enPuncts = ['.',',',';',':','-','\'','\'\'','``','!','?','/','"','[',']','(',')','{','}','$','%','@','#']
    others = ['>','<','=','--','&','|','please','thanks','hi','hello']
    
    if token in enStops or token in enPuncts or token in others or len(token)<2 or not token.isalpha():
        return False
    else:
        return True

def keep_it_strict1(token):
    """ This function returns whether a unigram is part of the stop words and special characters or not.
    If it is not part of stop words,enpuncts and others,Then return false else true.The stop words list 
    is imported from nltk.corpus. The difference between keep_it_strict and keep_it_strict1 is that the 
    cleaning list is quite different between summary and description. So keep_it_strict is used for cleaning
    summary and the latter is used for cleaning the description"""
    from nltk.corpus import stopwords
    enStops = stopwords.words('english')
    enPuncts = ['.',',',';',':','-','\'','\'\'','``','!','?','/','"','[',']','(',')','{','}','$','%','@','#']
    others = ['>','<','=','--','&','|','pleas','thank','hi','hello','request','need','id','request','attach','wa','ha','see','u','number','global','jci','thank','could','contact','n','thank','step','hi','new','http','na','get','inc','dear','egard','still','also','johnson','would','due','show','set','know','note','c','like','show','state','hour','cc','e','america','extjcicom','yes','ir','ii','j','r','st','phone','done','make','asap','jc']
    
    if token in enStops or token in enPuncts or token in others or len(token)<2 or not token.isalpha():
        return False
    else:
        return True
       
def get_words_stemmed(sents, keep_check = keep_it_strict, stemmer = nltk.stem.snowball.SnowballStemmer("english").stem):
    """ This function is to clean and stem the words of the summary or description in the ticket for assisting the prediction of tower process.
    Here sents is the summary or description of ticket, keep_check decides which function to be applied on summary or description and stemmer
    is the package for stemming the tokens.This function returns a output of tickets and the important(pruned,stemmed) words in the summary or 
    description of the tickets,the list of words in summary or description with the count and the details of where the words are found"""
    import nltk    
    #tokens_list = [[idx, nltk.word_tokenize(sents[idx])] for idx in sents.index]
    #tokens_list = [nltk.word_tokenize(sents[idx]) for idx in sents.index]
    tokens_list=[]
    for idx in sents.index:
        print(idx)
        if not pd.isnull(sents[idx]):
            new=nltk.word_tokenize(sents[idx])
            tokens_list.append(new)            
        else:
            tokens_list.append([])
            
    word_count = {}
    word_found_in = {}
    
    for i in range(len(tokens_list)):
        doc = []
        if not tokens_list[i]:
            continue
        else:
          for tok in tokens_list[i]:
            try:
                tok = stemmer(tok.lower())  # Stemming
            except UnicodeDecodeError:
                print(tok,i)
                continue
            
            if keep_check(tok):
                
                if tok in word_count.keys(): 
                    word_count[tok] +=1
                    word_found_in[tok].append(sents.index[i])
                else: 
                    word_count[tok] =1
                    word_found_in[tok] = [sents.index[i]]
            
                doc.append(tok)
               
            # Modifying the docs
            tokens_list[i] = doc
            
    #print(word_found_in)
        
    return pd.Series(tokens_list, index = sents.index), pd.DataFrame({'Word':word_count.keys(), 'Count':word_count.values()}, 
                                     columns = ['Word','Count']), word_found_in

def vocab_load():
    """This function provides the set of inbuilt resources required for the algorithm to predict the tower.
    Here A and B are the term affinity matrix for summary and description respectively and C and D are
    the vocabulary list of summary and description respectively"""
    A = pd.read_csv('term_summary.csv',index_col='Index')    
    B = pd.read_csv('term_description.csv',index_col='Index')
    C = pd.read_csv('vocab_summary.csv',index_col='Index')
    D = pd.read_csv('vocab_description.csv',index_col='Index')
    return A,B,C,D

def clean(data,col):
    """This function performs the initial cleaning steps to be performed on the raw data.It removes the all the unicode functions in the text
    of summary or description.Once the unicode functions are removed, the text is sentence tokenized for making them prepared for tokenization
    into individual words. Then the text is converted to lower case and few special characters amidst the words(#,/,-) are replaced with space(' ').
    Here data is the entire data to be processed and the col refers to summary or description on which initial cleaning needs to be performed"""
    sent_tokenizer=nltk.data.load('tokenizers/punkt/english.pickle')
    out=[]
    for i in data.index:    
        if not pd.isnull(data[col][i]):
            name=''.join([a if ord(a) < 128 else ' ' for a in data[col][i]])
            print(name)
            name2=' '.join(sent_tokenizer.tokenize(name))
            out.append([i,data['Key'][i],name2])        
        else:
            out.append([i,data['Key'][i],data[col][i]])       
    out=pd.DataFrame(out)
    out.columns=['Index','Key',col]
    out=out.set_index(['Index'])
    out.loc[:,col] = out[col].str.lower()
    out.loc[:,col] = out[col].str.replace('#', ' ')
    out.loc[:,col] = out[col].str.replace('-', ' ')
    out.loc[:,col] = out[col].str.replace('/', ' ')
    return out

def tower_possib(AA,BB,CC):
    """This function provides tower possibility for the matrix for the summary or description
    of the ticket.Here AA is the input data(summary or description of tickets),BB is the
    vocabulary list(summary or description) and CC is the term affinity matrix(summary or description)"""
    nTower=6
    result = {}
    for idx in AA.index:
        match = np.zeros(nTower)
        #print(summaries.loc[idx,'Pruned+Stemmed'])
        if AA.loc[idx,'Pruned+Stemmed']:   
            for term in AA.loc[idx,'Pruned+Stemmed']:
            #print(term)
                if term in BB.values:
                #print(term,match)
                    match = match + CC.loc[term]
            result[idx] = match/match.sum()
        else:
            result[idx]=match
    return result
    
def Parallel(tower_possibility,tower_possibility1,data):
    """This function predicts the tower for each ticket in parallel manner.It combines the tower possibility
    of summary and description of each ticket into one and then normalize it.Then the highest valus in the
    probability in the tower possibility is checked for minimal probability and confidence limit.If succeeds
    tower is predicted else "Dont Know" is print to highlight manual classification"""
    tower_poss = tower_possibility.add(tower_possibility1)
    tower_poss = tower_poss.div(tower_poss.sum(axis=1),axis=0)
    new=tower_poss.to_dict()
    from operator import itemgetter                                
    prediction=[]
    for i in data.index:
        print (i)
        keyval=[]
        for j in new.keys():
            v=new[j][i]
            keyval.append([j,v])
            keyval=sorted(keyval,key=itemgetter(1),reverse=True)
        highest=keyval[0][1]
        print(highest)
        second=keyval[1][1]
        print(second)
        confidence=highest-second
        print(confidence)
        if highest>=0.2 and confidence>=0.05:
            predict=keyval[0][0]
        else:
            predict="Dont Know"
        prediction.append([i,predict])    
    Prediction=pd.DataFrame(prediction)
    Prediction.columns=['Index','Prediction']
    Prediction=Prediction.set_index(['Index'])
    PredictionFinal=pd.concat([data[['Key','Summary','Description']],Prediction],axis=1)
    return PredictionFinal

def main(argv):
    """This function is the driver code for the entire algorithm. It gets the input file location
    and output file loaction. Then it process the input file with the help of other function in the module and
    predicts the tower for the tickets in the input file. The prediction output is then written on the output location
    provided"""
    import numpy as np
    import pandas as pd
    import re
    import nltk
    from nltk.corpus import wordnet as wn
    from nltk.stem import WordNetLemmatizer
    import sys,getopt
    inputfile = ''
    outputfile = ''
    try:
      opts, args = getopt.getopt(argv,"hi:o:",["ifile=","ofile="])
    except getopt.GetoptError:
      print ('test.py -i <inputfile> -o <outputfile>')
      sys.exit(2)
    for opt, arg in opts:
      if opt == '-h':
         print ('test.py -i <inputfile> -o <outputfile>')
         sys.exit()
      elif opt in ("-i", "--ifile"):
         inputfile = arg
      elif opt in ("-o", "--ofile"):
         outputfile = arg    
    data = pd.read_csv(inputfile)
    tickets = data[['Key', 'Summary', 'Description']]
    summaries=clean(tickets,'Summary')
    description=clean(tickets,'Description')
    summaries['Pruned+Stemmed'], summary_tokens_count, wfi1 =get_words_stemmed(summaries['Summary'])
    description['Pruned+Stemmed'], description_tokens_count, wfi3 =get_words_stemmed(description['Description'],keep_check = keep_it_strict1)    
    term_summary,term_description,vocab_summary,vocab_description=vocab_load()
    result = tower_possib(summaries,vocab_summary,term_summary)
    result1 = tower_possib(description,vocab_description,term_description)
    tower_possibility = pd.DataFrame(result).T
    tower_possibility1 = pd.DataFrame(result1).T    
    tower_possibility = tower_possibility.fillna(0)
    tower_possibility1 = tower_possibility1.fillna(0)
    Output=Parallel(tower_possibility,tower_possibility1,tickets)
    Output.to_csv(open(outputfile,'w'), index = False)    

if __name__=="__main__":
    main(sys.argv[1:])
    
#Output=main('C://Users//RahulSwami//Desktop//InputData2.csv','C://Users//RahulSwami//Desktop//term_summary.csv','C://Users//RahulSwami//Desktop//term_description.csv','C://Users//RahulSwami//Desktop//vocab_summary.csv','C://Users//RahulSwami//Desktop//vocab_description.csv',Combined,'C://Users//RahulSwami//Desktop//Module_Output.csv')

