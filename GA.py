#from MetaHeuristicsFS import FeatureSelection

from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score,precision_score,recall_score,accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import auc,roc_auc_score,average_precision_score
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import StratifiedKFold
from sklearn.utils import compute_class_weight

from sklearn import svm

from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.util import ngrams

import pandas as pd
import numpy as np
import joblib

import warnings
from collections import Counter
import random as rd
import time
import gc
import pickle
import sys
import os
import re

class TextFeatureSelectionGA():
    '''Use genetic algorithm for selecting text tokens which give best classification results
    
    Genetic Algorithm Parameters
    ----------
    
    generations : Number of generations to run genetic algorithm. 500 as deafult, as used in the original paper
    
    population : Number of individual chromosomes. 50 as default, as used in the original paper
    
    prob_crossover : Probability of crossover. 0.9 as default, as used in the original paper
    
    prob_mutation : Probability of mutation. 0.1 as default, as used in the original paper
    
    percentage_of_token : Percentage of word features to be included in a given chromosome.
        50 as default, as used in the original paper.
    runtime_minutes : Number of minutes to run the algorithm. This is checked in between generations.
        At start of each generation it is checked if runtime has exceeded than alloted time.
        If case run time did exceeds provided limit, best result from generations executed so far is given as output.
        Default is 2 hours. i.e. 120 minutes.
        
    References
    ----------
    Noria Bidi and Zakaria Elberrichi "Feature Selection For Text Classification Using Genetic Algorithms"
    https://ieeexplore.ieee.org/document/7804223
    
    '''
    
    def __init__(self,generations=20,population=50,prob_crossover=0.9,prob_mutation=0.1,percentage_of_token=50,runtime_minutes=1):
        self.generations=generations
        self.population=population
        self.prob_crossover=prob_crossover
        self.prob_mutation=prob_mutation
        self.percentage_of_token=percentage_of_token
        self.runtime_minutes=runtime_minutes
        
    def _cost_function_value(self,y_test,y_test_pred,cost_function,avrg):
        if cost_function == 'f1':
            if avrg == 'binary':
                metric=f1_score(y_test,y_test_pred,average='binary')

        elif cost_function == 'precision':
            if avrg == 'binary':
                metric=precision_score(y_test,y_test_pred,average='binary')

        elif cost_function == 'recall':
            if avrg == 'binary':
                metric=recall_score(y_test,y_test_pred,average='binary')
        elif cost_function == 'accuracy':
                metric=accuracy_score(y_test,y_test_pred)

        return metric


    def _computeFitness(self,gene,unique_words,x,y,model,model_metric,avrg,analyzer,min_df,max_df,stop_words,tokenizer,token_pattern,lowercase):
        ### create tfidf matrix for only terms which are in gene
        # get terms from gene and vocabulary combnation
        term_to_use=list(np.array(unique_words)[list(map(bool,gene))])

        metric_result=[]
        skfold=StratifiedKFold(n_splits=5)

        ##get words based on gene index to get vocabulary
        term_to_use=list(np.array(unique_words)[list(map(bool,gene))])

        for train_index, test_index in skfold.split(x,y):
            #get x_train,y_train  x_test,y_test
            X_train, X_test = list(np.array(x)[train_index]),list(np.array(x)[test_index]) 
            y_train, y_test = np.array(y)[train_index],np.array(y)[test_index]

            ##based on vocabulary set, create tfidf matrix for train and test data
            tfidf=TfidfVectorizer(vocabulary=term_to_use,analyzer=analyzer,min_df=min_df,max_df=max_df,stop_words=stop_words,tokenizer=tokenizer,token_pattern=token_pattern,lowercase=lowercase)
            tfidfvec_vectorizer=tfidf.fit(X_train)

            #get x train and test
            X_train=tfidfvec_vectorizer.transform(X_train)
            X_test=tfidfvec_vectorizer.transform(X_test)

            #train model
            model.fit(X_train,y_train)

            #predict probability for test
            y_test_pred=model.predict(X_test)

            #get desired metric and append to metric_result
            metric_result.append(self._cost_function_value(y_test,y_test_pred,model_metric,avrg))

        return np.mean(metric_result)



    def _check_unmatchedrows(self,population_matrix,population_array):
        pop_check=0
        #in each row of population matrix
        for pop_so_far in range(population_matrix.shape[0]):
            #check if it is duplicate
            if sum(population_matrix[pop_so_far]!=population_array)==population_array.shape[0]:
                #assign 1 as value if it is duplicate and break loop
                pop_check=1
                break

        return pop_check

    def _get_population(self,population,population_matrix,population_array):
        iterate=0
        ##append until population and no duplicate chromosome
        while population_matrix.shape[0] < population:
            ##prepare population matrix
            rd.shuffle(population_array)
            #check if it is first iteration, if yes append
            if iterate==0:
                population_matrix = np.vstack((population_matrix,population_array))
                iterate+=1
            #if second iteration and one chromosome already, check if it is duplicate
            elif iterate==1 and sum(population_matrix[0]==population_array)!=population_array.shape[0]:
                population_matrix = np.vstack((population_matrix,population_array))
                iterate+=1
            #when iteration second and beyond check duplicacy
            elif iterate>1:# and self._check_unmatchedrows(population_matrix,population_array)==0:
                population_matrix = np.vstack((population_matrix,population_array))
                iterate+=1

        return population_matrix


    def _get_parents(self,population_array,population_matrix,unique_words,x,y,model,model_metric,avrg,analyzer,min_df,max_df,stop_words,tokenizer,token_pattern,lowercase):

        #keep space for best chromosome
        parents = np.empty((0,population_array.shape[0]))

        #get 6 unique index to fetch from population
        indexes=np.random.randint(0,population_matrix.shape[0],6)
        while len(np.unique(indexes))<6:
            indexes=np.random.randint(0,len(population_matrix),6)

        #mandatory run twice as per GA algorithm
        for run_range in range(2):
            #get 3 unique index to fetch from population
            #if first run then until half
            if run_range==0:
                index_run=indexes[0:3]
            #if second run then from half till end
            else:
                index_run=indexes[3:]

            ##gene pool 1
            gene_1 = population_matrix[index_run[0]]
            #cost of gene 1
            cost1=self._computeFitness(gene=gene_1,unique_words=unique_words,x=x,y=y,model=model,model_metric=model_metric,avrg=avrg,analyzer=analyzer,min_df=min_df,max_df=max_df,stop_words=stop_words,tokenizer=tokenizer,token_pattern=token_pattern,lowercase=lowercase)
            ##gene pool 2
            gene_2 = population_matrix[index_run[1]]
            #cost of gene 2
            cost2=self._computeFitness(gene=gene_2,unique_words=unique_words,x=x,y=y,model=model,model_metric=model_metric,avrg=avrg,analyzer=analyzer,min_df=min_df,max_df=max_df,stop_words=stop_words,tokenizer=tokenizer,token_pattern=token_pattern,lowercase=lowercase)
            ##gene pool 3
            gene_3 = population_matrix[index_run[2]]
            #cost of gene 3
            cost3=self._computeFitness(gene=gene_3,unique_words=unique_words,x=x,y=y,model=model,model_metric=model_metric,avrg=avrg,analyzer=analyzer,min_df=min_df,max_df=max_df,stop_words=stop_words,tokenizer=tokenizer,token_pattern=token_pattern,lowercase=lowercase)

            #get best chromosome from 3 and assign best chromosome.
            if cost1==max(cost1,cost2,cost3):
                parents = np.vstack((parents,gene_1))
            elif cost2==max(cost1,cost2,cost3):
                parents = np.vstack((parents,gene_2))
            elif cost3==max(cost1,cost2,cost3):
                parents = np.vstack((parents,gene_3))

        #get 2 best chromosome identified as parents
        return parents[0],parents[1]

    def _crossover(self,parent1,parent2,prob_crossover):

        #placeholder for child chromosome
        child1 = np.empty((0,len(parent1)))
        child2 = np.empty((0,len(parent2)))

        #generate random number ofr crossover probability
        crsvr_rand_prob = np.random.rand()

        ## if random decimal generated is less than probability of crossover set
        if crsvr_rand_prob < prob_crossover:
            index1 = np.random.randint(0,len(parent1))
            index2 = np.random.randint(0,len(parent1))

            # get different indices
            # to make sure you will crossover at least one gene
            while index1 == index2:
                index2 = np.random.randint(0,len(parent1))

            index_parent1 = min(index1,index2) 
            index_parent2 = max(index1,index2) 

            ## Parent 1
            # first segment
            first_seg_parent1 = parent1[:index_parent1]
            # middle segment; where the crossover will happen
            mid_seg_parent1 = parent1[index_parent1:index_parent2+1]
            # last segment
            last_seg_parent1 = parent1[index_parent2+1:]
            ## child from all segments
            child1 = np.concatenate((first_seg_parent1,mid_seg_parent1,last_seg_parent1))                

            ### Parent 2
            # first segment
            first_seg_parent2 = parent2[:index_parent2]
            # middle segment; where the crossover will happen
            mid_seg_parent2 = parent2[index_parent2:index_parent2+1]
            # last segment
            last_seg_parent2 = parent2[index_parent2+1:]
            ## child from all segments
            child2 = np.concatenate((first_seg_parent2,mid_seg_parent2,last_seg_parent2))        
            return child1,child2
        #if probability logic is bypassed
        else:
            return parent1,parent2

    def _mutation(self,child,prob_mutation):

        # mutated child 1 placeholder
        mutated_child = np.empty((0,len(child)))

        ## get random probability at each index of chromosome and start with 0    
        t = 0
        for cld1 in child:
            rand_prob_mutation = np.random.rand() # do we mutate or no???
            # if random decimal generated is less than random probability, then swap value at index position
            if rand_prob_mutation < prob_mutation:
                # swap value
                if child[t] == 0:
                    child[t] = 1            
                else:
                    child[t] = 0
                # assign temporary child chromosome
                mutated_child = child

            #if random prob is >= mutation probability, assign as it is
            else:
                mutated_child = child

            # increase counter
            t = t+1
        return mutated_child
    
    def _getPopulationAndMatrix(self,doc_list,label_list,analyzer,min_df,max_df,stop_words,tokenizer,token_pattern,lowercase):
        #get null free df
        temp_df=pd.DataFrame({'doc_list':doc_list,'label_list':label_list})
        temp_df=temp_df[(~temp_df['doc_list'].isna()) & (~temp_df['label_list'].isna())]
        temp_df.reset_index(inplace=True,drop=True)
        label_list=temp_df['label_list'].tolist()
        doc_list=temp_df['doc_list'].tolist()
        del temp_df
        gc.collect()

        #get unique tokens
        tfidfvec = TfidfVectorizer(analyzer=analyzer,min_df=min_df,max_df=max_df,stop_words=stop_words,tokenizer=tokenizer,token_pattern=token_pattern,lowercase=lowercase)
        tfidfvec_vectorizer = tfidfvec.fit(doc_list)
        unique_words=list(tfidfvec_vectorizer.vocabulary_.keys())

        #count of tokens to consider based on percentage
        chromosome_to_feature = int(round((len(unique_words)/100)*self.percentage_of_token))

        #generate chromosome with number of 1 equal to percentage from total features specified by user
        population_array=np.concatenate([np.zeros(len(unique_words)-chromosome_to_feature),np.ones(chromosome_to_feature)])
        #shuffle after concatenating 0 and 1
        rd.shuffle(population_array)

        #create blank population matrix to append all individual chromosomes. number of rows equal to population size
        population_matrix = np.empty((0,len(unique_words)))

        #get population matrix
        population_matrix=self._get_population(self.population,population_matrix,population_array)

        #best solution for each generation
        best_of_a_generation = np.empty((0,len(population_array)+1))
        
        return doc_list,label_list,unique_words,population_array,population_matrix,best_of_a_generation

    def getGeneticFeatures(self,doc_list,label_list,model=LogisticRegression(),model_metric='accuracy',avrg='binary',analyzer='word',min_df=2,max_df=1.0,stop_words=None,tokenizer=None,token_pattern='(?u)\\b\\w\\w+\\b',lowercase=True):
        '''
        Data Parameters
        ----------        
        doc_list : text documents in a python list. 
            Example: ['i had dinner','i am on vacation','I am happy','Wastage of time']
        
        label_list : labels in a python list.
            Example: ['Neutral','Neutral','Positive','Negative']
        
        
        Modelling Parameters
        ----------
        model : Set a model which has .fit function to train model and .predict function to predict for test data. 
            This model should also be able to train classifier using TfidfVectorizer feature.
            Default is set as Logistic regression in sklearn
        
        model_metric : Classifier cost function. Select one from: ['f1','precision','recall'].
            Default is F1
        
        avrg : Averaging used in model_metric. Select one from ['micro', 'macro', 'samples','weighted', 'binary'].
            For binary classification, default is 'binary' and for multi-class classification, default is 'micro'.
        
        
        TfidfVectorizer Parameters
        ----------
        analyzer : {'word', 'char', 'char_wb'} or callable, default='word'
            Whether the feature should be made of word or character n-grams.
            Option 'char_wb' creates character n-grams only from text inside
            word boundaries; n-grams at the edges of words are padded with space.
            
        min_df : float or int, default=2
            When building the vocabulary ignore terms that have a document
            frequency strictly lower than the given threshold. This value is also
            called cut-off in the literature.
            If float in range of [0.0, 1.0], the parameter represents a proportion
            of documents, integer absolute counts.
            This parameter is ignored if vocabulary is not None.
        max_df : float or int, default=1.0
            When building the vocabulary ignore terms that have a document
            frequency strictly higher than the given threshold (corpus-specific
            stop words).
            If float in range [0.0, 1.0], the parameter represents a proportion of
            documents, integer absolute counts.
            This parameter is ignored if vocabulary is not None.
        stop_words : {'english'}, list, default=None
            If a string, it is passed to _check_stop_list and the appropriate stop
            list is returned. 'english' is currently the only supported string
            value.
            There are several known issues with 'english' and you should
            consider an alternative (see :ref:`stop_words`).
            If a list, that list is assumed to contain stop words, all of which
            will be removed from the resulting tokens.
            Only applies if ``analyzer == 'word'``.
            If None, no stop words will be used. max_df can be set to a value
            in the range [0.7, 1.0) to automatically detect and filter stop
            words based on intra corpus document frequency of terms.
        tokenizer : callable, default=None
            Override the string tokenization step while preserving the
            preprocessing and n-grams generation steps.
            Only applies if ``analyzer == 'word'``
        token_pattern : str, default=r"(?u)\\b\\w\\w+\\b"
            Regular expression denoting what constitutes a "token", only used
            if ``analyzer == 'word'``. The default regexp selects tokens of 2
            or more alphanumeric characters (punctuation is completely ignored
            and always treated as a token separator).
            If there is a capturing group in token_pattern then the
            captured group content, not the entire match, becomes the token.
            At most one capturing group is permitted.
        lowercase : bool, default=True
            Convert all characters to lowercase before tokenizing.        
        '''
        
        start = time.time()
        
        #define cost function averaging
        if len(set(label_list))>2:
            avrg='micro'
        else:
            avrg='binary'
        
        #get all parameters needed for GA
        doc_list,label_list,unique_words,population_array,population_matrix,best_of_a_generation=self._getPopulationAndMatrix(doc_list,label_list,analyzer=analyzer,min_df=min_df,max_df=max_df,stop_words=stop_words,tokenizer=tokenizer,token_pattern=token_pattern,lowercase=lowercase)
        
        #Execute GA
        for genrtn in range(self.generations):
            
            ##if time exceeds then break loop
            if (time.time()-start)//60 > self.runtime_minutes:
                print('Run time exceeded allocated time. Producing best features generated so far:')
                break
            
            # placeholder for saving the new generation
            new_population = np.empty((0,len(population_array)))

            # placeholder for saving the new generation and obj func val
            new_population_with_obj_val = np.empty((0,len(population_array)+1))

            # placeholder for saving the best solution for each generation
            sorted_best = np.empty((0,len(population_array)+1))

            ## generate new set of population in each generation
            # each iteration gives 2 chromosome.
            # Doing it half the population size will mean getting matrix of population size equal to original matrix
            for family in range(int(self.population/2)):
                #get parents
                parent1,parent2=self._get_parents(population_array=population_array,population_matrix=population_matrix,unique_words=unique_words,x=doc_list,y=label_list,model=model,model_metric=model_metric,avrg=avrg,analyzer=analyzer,min_df=min_df,max_df=max_df,stop_words=stop_words,tokenizer=tokenizer,token_pattern=token_pattern,lowercase=lowercase)

                #crossover
                child1,child2=self._crossover(parent1=parent1,parent2=parent2,prob_crossover=self.prob_crossover)

                #mutation
                mutated_child1=self._mutation(child=child1,prob_mutation=self.prob_mutation)
                mutated_child2=self._mutation(child=child2,prob_mutation=self.prob_mutation)

                #get cost function for 2 mutated child and print for generation, family and child
                cost1=self._computeFitness(gene=mutated_child1,unique_words=unique_words,x=doc_list,y=label_list,model=model,model_metric=model_metric,avrg=avrg,analyzer=analyzer,min_df=min_df,max_df=max_df,stop_words=stop_words,tokenizer=tokenizer,token_pattern=token_pattern,lowercase=lowercase)
                cost2=self._computeFitness(gene=mutated_child2,unique_words=unique_words,x=doc_list,y=label_list,model=model,model_metric=model_metric,avrg=avrg,analyzer=analyzer,min_df=min_df,max_df=max_df,stop_words=stop_words,tokenizer=tokenizer,token_pattern=token_pattern,lowercase=lowercase)

                #create population for next generaion
                new_population = np.vstack((new_population,mutated_child1,mutated_child2))

                #save cost and child
                mutant1_with_obj_val = np.hstack((cost1,mutated_child1))
                mutant2_with_obj_val = np.hstack((cost2,mutated_child2))
                #stack both chromosome of the family
                new_population_with_obj_val = np.vstack((new_population_with_obj_val,mutant1_with_obj_val,mutant2_with_obj_val))

            #at end of the generation, change population as the stacked chromosome set from previous generation
            population_matrix=new_population

            ### find best solution for generation based on objective function and stack
            sorted_best = np.array(sorted(new_population_with_obj_val,key=lambda x:x[0],reverse=True))

            # print and stack
            print('Generation:',genrtn,'best score',sorted_best[0][0])
            best_of_a_generation = np.vstack((best_of_a_generation,sorted_best[0]))

        #sort by metric
        best_metric_chromosome_pair = np.array(sorted(best_of_a_generation,key=lambda x:x[0],reverse=True))[0]

        #best chromosome, metric and vocabulary
        best_chromosome=best_metric_chromosome_pair[1:]

        best_metric=best_metric_chromosome_pair[0]
        print('Best metric:',best_metric)

        best_vocabulary=list(np.array(unique_words)[list(map(bool,best_chromosome))])
        return best_vocabulary   
      
      import pandas as pd
data=pd.read_excel("/content/drive/MyDrive/AI/eval.xlsx")
data

doc_list=data['text']
label_list=data['HS']
label_list

for i in range(len(label_list)):
     if label_list[i]==1:
         label_list[i] = 'pos'
     else:
         label_list[i] = 'neg'
label_list

getGAobj=TextFeatureSelectionGA(percentage_of_token=60)
best_vocabulary=getGAobj.getGeneticFeatures(doc_list=doc_list,label_list=label_list)
