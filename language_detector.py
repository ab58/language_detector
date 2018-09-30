# -*- coding: utf-8 -*-
"""
Created on Wed Aug  8 13:48:48 2018

@author: arjunb
"""

import os, sys, time, operator, pprint
import numpy as np
import pandas as pd
from sklearn import linear_model, naive_bayes, neighbors, svm
from sklearn.feature_extraction import text
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
import nltk
from nltk.tokenize import sent_tokenize



def get_ngrams(directory, data_file, data_type, n, stat_file):
    
    if (data_type != "word" and data_type != "char"):
        print("Invalid data type: please enter either \"word\" or \"char\"")
        sys.exit(0)
    
    
    #print("Gathering "+data_type+" "+str(n)+"-grams for file "+data_file+"\n")
    f = open(directory+"/"+data_file, encoding="latin1")
    file_contents = f.read()
    #print(file_contents)

    #we need to get counts of all ngrams
    ngram_counts = {}
    joiner = ""
    if data_type == "word":
        file_contents = file_contents.split()
        joiner = " "
    
    for i in range(len(file_contents)-n+1):
        ngram = joiner.join(file_contents[i:i+n])
        
        if "\t" in ngram or "\n" in ngram or any(char.isdigit() for char in ngram):
            continue
        
        
        #print("current ngram is "+ngram)
        if ngram not in ngram_counts:
            ngram_counts[ngram] = 0
        ngram_counts[ngram] += 1
    
    ngram_counts = sorted(ngram_counts.items(), key=operator.itemgetter(1), reverse=True)
    stat_file.write("Top 10 "+data_type+" "+str(n)+"-grams for "+data_file+":\n\n")
    stat_file.write(pprint.pformat(ngram_counts[:10])+"\n\n")
    stat_file.write("Total number of unique "+data_type+" "+str(n)+"-grams for "+data_file+": "+str(len(ngram_counts))+"\n\n")
    
    f.close()
    
    #all_ngrams.update([i[0] for i in ngram_counts[:10000]])
    
    return ngram_counts#[:10000]
    

def append_dict(dict1, dict2):
    
    for k, v in dict2:
        if k not in dict1:
            dict1[k] = 0
        dict1[k] += v
    

def model_result(model, X_train, y_train, X_test, y_test):
    
    model.fit(X_train, y_train)
    
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    return test_pred[0]


def process_and_classify(all_ngrams, ngram_counts_test, test_str, truelabel, X_train, y_train):#, feat_sel):
    
    tic = time.time()
    print("processing..........", end="")
    
    for ngram in all_ngrams:
        ngram_counts_test[ngram] = test_str.count(ngram)
        
    one_hot_vector_lbl = {truelabel : np.array(list(ngram_counts_test.values()))}
        
    X_test = pd.DataFrame.from_dict(one_hot_vector_lbl, orient="index")
    y_test = X_test.index
    #print("\n")
    #print(X_train.shape)
    #print(X_test.shape)
    #X_test = feat_sel.transform(X_test)
    #print(X_test.shape)
    
        
    #test_prediction = model_result(linear_model.LogisticRegression(C=1), X_train, y_train, X_test, y_test)
    test_prediction = model_result(naive_bayes.MultinomialNB(), X_train, y_train, X_test, y_test)
    
    toc = time.time()
    print(str(int(toc-tic))+"s")
    
    if truelabel not in y_train:
        print("True language not in training data: closest predicted language is "+test_prediction)
    else:
        print("This language is predicted to be "+test_prediction)
    if test_prediction == truelabel:
        print("Correct!")
    else:
        print("Incorrect")
    
    return test_prediction
        

def main():
    
    tic_all = time.time()
    print("\nTHE LANGUAGE DETECTOR")
    
    directory = sys.argv[1]
    all_ngrams = set([])
    stat_file = open(sys.argv[2], "w")
    ngram_counts_all_langs = {}
    
    tic = time.time()
    print("Gathering training data for all languages......", end="")
    
    all_sentences = {}
    for data_file in os.listdir(directory):
        
        language = data_file.split(".")[1]
        f = open(directory+"/"+data_file, encoding="latin1")
        full_text = f.read()
        sentences = sent_tokenize(full_text)
        for sent in sentences:
            all_sentences[sent] = language

        combined_ngram_counts = {}
        if language in ngram_counts_all_langs:
            combined_ngram_counts = ngram_counts_all_langs[language]
        
        w1grams = get_ngrams(directory, data_file, "word", 1, stat_file)
        append_dict(combined_ngram_counts, w1grams)
        #w2grams = get_ngrams(directory, data_file, "word", 2, stat_file)
        #append_dict(combined_ngram_counts, w2grams)
        #w3grams = get_ngrams(directory, data_file, "word", 3, stat_file)
        #append_dict(combined_ngram_counts, w3grams)
        #w4grams = get_ngrams(directory, data_file, "word", 4, stat_file)
        #append_dict(combined_ngram_counts, w4grams)
        #w5grams = get_ngrams(directory, data_file, "word", 5, stat_file)
        #append_dict(combined_ngram_counts, w5grams)
        
        c1grams = get_ngrams(directory, data_file, "char", 1, stat_file)
        append_dict(combined_ngram_counts, c1grams)
        c2grams = get_ngrams(directory, data_file, "char", 2, stat_file)
        append_dict(combined_ngram_counts, c2grams)
        c3grams = get_ngrams(directory, data_file, "char", 3, stat_file)
        append_dict(combined_ngram_counts, c3grams)
        c4grams = get_ngrams(directory, data_file, "char", 4, stat_file)
        append_dict(combined_ngram_counts, c4grams)
        c5grams = get_ngrams(directory, data_file, "char", 5, stat_file)
        append_dict(combined_ngram_counts, c5grams)
        
        
        combined_ngram_counts = dict(sorted(combined_ngram_counts.items(), key=operator.itemgetter(1), reverse=True))
        ngram_counts_all_langs[language] = combined_ngram_counts
        all_ngrams.update(combined_ngram_counts.keys())
        
    toc = time.time()
    print(str(int(toc-tic))+"s")
    
    for lang in ngram_counts_all_langs.keys():    
        stat_file.write("Total unique ngrams for language "+lang+": "+str(len(ngram_counts_all_langs[lang]))+"\n\n")
    
    
    stat_file.write("Total unique ngrams ocurring in all languages: "+str(len(all_ngrams))+"\n\n")
    stat_file.write (str(sorted(all_ngrams)))
    
    ngram_counts_one_hot = {}
    for lang in ngram_counts_all_langs.keys():
        ngram_counts_one_hot[lang] = {}
        '''
        we want to put all items in all_ngrams into ngram_counts_one_hot[lang].
        to do this, we go through each item here, and check it against what's
        in ngram_counts_all_langs[lang]. If the item is found, add this item 
        plus its count value to ngram_counts_one_hot[lang]. If it's not found,
        add the item with a count of 0 to ngram_counts_one_hot[lang].
        '''
        for ngram in all_ngrams:
            if ngram in ngram_counts_all_langs[lang]:
                ngram_counts_one_hot[lang][ngram] = ngram_counts_all_langs[lang][ngram]
            else:
                ngram_counts_one_hot[lang][ngram] = 0
    
    one_hot_vectors_raw = {}
    for lang in ngram_counts_one_hot.keys():
        #raw_count = ngram_counts_one_hot[lang]["os"]
        #relative_weight =  raw_count / len(ngram_counts_all_langs[lang]) * 100
        #print(lang+": "+str(raw_count))
        #print(lang+": "+str(relative_weight))
        
        one_hot_vectors_raw[lang] = ngram_counts_one_hot[lang].values()
        one_hot_vectors_raw[lang] = np.array(list(one_hot_vectors_raw[lang]))
        
    #np.set_printoptions(threshold=np.inf)
    
    #print(one_hot_vectors_raw)
    #print(ngram_counts_one_hot["ES"])
    
    
    stat_file.close()
    

    X_train = pd.DataFrame.from_dict(one_hot_vectors_raw, orient="index")
    y_train = X_train.index
    
    feat_sel = VarianceThreshold(0.5)
    #X_train = feat_sel.fit_transform(X_train)
    
    print("\nSupported languages:")
    print(" ".join([lg for lg in list(y_train)]))
    
    
    '''
    To get test data, put the testing string into a one-hot vector with all_ngrams
    '''
    
    print("\nTesting for single file line-by-line")
    results_file = open("language_detector_results_report.txt", "w", encoding="utf-8")
    results_file.write("RESULTS REPORT: Line-by-Line Test File")
    line_by_line_file = open(sys.argv[4])
    lines = line_by_line_file.readlines()
    
    results_train = open("language_detector_training_results.txt", "w", encoding="utf-8")
    results_train.write("RESULTS REPORT: Training Data")
    correct_predictions_train = 0
    total_predictions_train = 0
    
    failure_counts_train = {}
    true_positives_train = {}
    
    for sent in all_sentences:
        truelabel = all_sentences[sent]
        if truelabel not in true_positives_train:
            true_positives_train[truelabel] = 0
        print("Training sentence: "+sent)
        results_train.write("\n==============================")
        results_train.write("\nTraining sentence: "+sent)
        results_train.write("\n\nTruth: " +truelabel)
        
        ngram_counts_train = {}
        prediction = process_and_classify(all_ngrams, ngram_counts_train, sent, truelabel, X_train, y_train)
        total_predictions_train += 1
        results_train.write("\nHyp: "+prediction)
        
        if prediction == truelabel:
            correct_predictions_train += 1
            true_positives_train[truelabel] += 1
            results_train.write("\nCorrect!")
        else:
            failure = truelabel+"-->"+prediction
            if failure not in failure_counts_train.keys():
                failure_counts_train[failure] = 0
            failure_counts_train[failure] += 1
            results_train.write("\nIncorrect")
        
        print("\nCorrect predictions so far: "+str(correct_predictions_train))
        print("Incorrect predictions so far: "+str(total_predictions_train-correct_predictions_train))
        print("\nfailure_counts so far: "+str(failure_counts_train))
        results_train.write("\n==============================")
    
    acc_train = correct_predictions_train / total_predictions_train
    
    correct_predictions_ll = 0
    total_predictions_ll = 0
    
    failure_counts = {}
    true_positives = {}
    '''
    the failure_counts dictionary will track all failures, with keys in the format "L1-->L2",
    and values as the number of times each particular failure occurred
    '''
    c = 0
    for line in lines:
        
        #if c == 5:
        #    break
        #c += 1
        
        truelabel = line.split("\t")[0]
        if truelabel not in true_positives:
            true_positives[truelabel] = 0
            
        test_str = line.split("\t")[1]
        print("\nTest string: "+test_str)
        results_file.write("\n==============================")
        results_file.write("\nTest string: "+test_str)
        results_file.write("\nTruth: " +truelabel)
        
        ngram_counts_lbl = {}
        prediction = process_and_classify(all_ngrams, ngram_counts_lbl, test_str, truelabel, X_train, y_train)#, feat_sel):
        total_predictions_ll += 1
        results_file.write("\nHyp: "+prediction)
        
        
        if prediction == truelabel:
            correct_predictions_ll += 1
            true_positives[truelabel] += 1
            results_file.write("\nCorrect!")
        else:
            failure = truelabel+"-->"+prediction
            if failure not in failure_counts.keys():
                failure_counts[failure] = 0
            failure_counts[failure] += 1
            results_file.write("\nIncorrect")
            
        print("\nCorrect predictions so far: "+str(correct_predictions_ll))
        print("Incorrect predictions so far: "+str(total_predictions_ll-correct_predictions_ll))
        print("\nfailure_counts so far: "+str(failure_counts))
        results_file.write("\n==============================")
          
    acc_line_by_line = correct_predictions_ll / total_predictions_ll
    
    
    print("\nTesting for full files in test data")
    
    correct_predictions_ff = 0
    total_predictions_ff = 0
    
    for test_filename in os.listdir(sys.argv[3]):
        
        print("\nLooking at file "+test_filename+"\n")
        test_file = open(sys.argv[3]+"/"+test_filename)
        test_str = test_file.read()
        truelabel = test_filename.split(".")[1]
        ngram_counts_test = {}
        prediction = process_and_classify(all_ngrams, ngram_counts_test, test_str, truelabel, X_train, y_train)#, feat_sel):
        total_predictions_ff += 1
        
        if prediction == truelabel:
            correct_predictions_ff += 1
            
    acc_full_files = correct_predictions_ff / total_predictions_ff
    
    
    emittors = {}
    attractors = {}
    emittors_train = {}
    attractors_train = {}

    for failure in failure_counts:
       emittor = failure.split("-->")[0]
       attractor = failure.split("-->")[1]
       
       if emittor not in emittors:
           emittors[emittor] = 0
       emittors[emittor] += failure_counts[failure]
       
       if attractor not in attractors:
           attractors[attractor] = 0
       attractors[attractor] += failure_counts[failure]
      
    for failure in failure_counts_train:
       emittor = failure.split("-->")[0]
       attractor = failure.split("-->")[1]
       
       if emittor not in emittors_train:
           emittors_train[emittor] = 0
       emittors_train[emittor] += failure_counts_train[failure]
       
       if attractor not in attractors_train:
           attractors_train[attractor] = 0
       attractors_train[attractor] += failure_counts_train[failure]
    
    
    metrics = {}   
    for label in true_positives:
        metrics[label] = [0, 0, 0]
        if label not in emittors:
            metrics[label][1] = 1.0
        else:
            metrics[label][1] = true_positives[label] / (true_positives[label] + emittors[label])
            
        if label not in attractors:
            metrics[label][0] = 1.0
        else:
            metrics[label][0] = true_positives[label] / (true_positives[label] + attractors[label])
        
        metrics[label][2] = 2*(metrics[label][0]*metrics[label][1] / (metrics[label][0]+metrics[label][1]))
        
    metrics_train = {}   
    for label in true_positives_train:
        metrics_train[label] = [0, 0, 0]
        if label not in emittors_train:
            metrics_train[label][1] = 1.0
        else:
            metrics_train[label][1] = true_positives_train[label] / (true_positives_train[label] + emittors_train[label])
            
        if label not in attractors_train:
            metrics_train[label][0] = 1.0
        else:
            metrics_train[label][0] = true_positives_train[label] / (true_positives_train[label] + attractors_train[label])
        
        metrics_train[label][2] = 2*(metrics_train[label][0]*metrics_train[label][1] / (metrics_train[label][0]+metrics_train[label][1]))
    
    failure_counts = sorted(failure_counts.items(), key=operator.itemgetter(1), reverse=True)  
    emittors = sorted(emittors.items(), key=operator.itemgetter(1), reverse=True)  
    attractors = sorted(attractors.items(), key=operator.itemgetter(1), reverse=True)
    failure_counts_train = sorted(failure_counts_train.items(), key=operator.itemgetter(1), reverse=True)  
    emittors_train = sorted(emittors_train.items(), key=operator.itemgetter(1), reverse=True)  
    attractors_train = sorted(attractors_train.items(), key=operator.itemgetter(1), reverse=True)
    
    
    print("\nAccuracy for training sentences: "+str(acc_train))
    print("\nAccuracy for line-by-line testing: "+str(acc_line_by_line))
    print("\nAccuracy for full file testing: "+str(acc_full_files))  
     
    
    results_file.write("\n\nMost frequent emittors (truth values in error):")
    for em in emittors:
        results_file.write("\n"+str(em[0])+": "+str(em[1]))
    results_file.write("\n\nMost frequent attractors (hyp values in error):")
    for at in attractors:
        results_file.write("\n"+str(at[0])+": "+str(at[1]))
    results_file.write("\n\nMost frequent failures (emittor-->attractor pairs):")
    for fail in failure_counts:
        results_file.write("\n"+str(fail[0])+": "+str(fail[1]))
    
    results_file.write("\n\nPrecision, Recall, and F-Scores:")
    results_file.write("\nLANG\tP\tR\tF")
    for lang in metrics:
        results_file.write("\n"+lang+"\t"+str(metrics[lang][0])+"\t"+str(metrics[lang][1])+"\t"+str(metrics[lang][2]))
    
    results_file.write("\n\nTotal testing examples: "+str(total_predictions_ll))
    results_file.write("\nTotal correct: "+str(correct_predictions_ll))
    results_file.write("\nTotal failures: "+str(total_predictions_ll-correct_predictions_ll))
    results_file.write("\nOverall Accuracy for line-by-line testing: "+str(acc_line_by_line)+"\n")
    
    
    results_train.write("\n\nMost frequent emittors (truth values in error):")
    for em in emittors_train:
        results_train.write("\n"+str(em[0])+": "+str(em[1]))
    results_train.write("\n\nMost frequent attractors (hyp values in error):")
    for at in attractors_train:
        results_train.write("\n"+str(at[0])+": "+str(at[1]))
    results_train.write("\n\nMost frequent failures (emittor-->attractor pairs):")
    for fail in failure_counts_train:
        results_train.write("\n"+str(fail[0])+": "+str(fail[1]))
    
    results_train.write("\n\nPrecision, Recall, and F-Scores:")
    results_train.write("\nLANG\tP\tR\tF")
    for lang in metrics_train:
        results_train.write("\n"+lang+"\t"+str(metrics_train[lang][0])+"\t"+str(metrics_train[lang][1])+"\t"+str(metrics_train[lang][2]))
    
    results_train.write("\n\nTotal training examples: "+str(total_predictions_train))
    results_train.write("\nTotal correct: "+str(correct_predictions_train))
    results_train.write("\nTotal failures: "+str(total_predictions_train-correct_predictions_train))
    results_train.write("\nOverall Training Accuracy: "+str(acc_train)+"\n")
    
    
    toc_all = time.time()
    running_time = toc_all - tic_all
    hours = running_time // 3600
    running_time %= 3600
    minutes = running_time // 60
    running_time %= 60
    seconds = int(running_time)
    print("Total Running Time: %d hours, %d minutes, %d seconds." % (hours, minutes, seconds))
    results_file.write("Total Running Time: %d hours, %d minutes, %d seconds." % (hours, minutes, seconds))
    results_file.close()
    results_train.close()

    

    
    


if __name__ == "__main__":
    main()
    
    
    
    
    