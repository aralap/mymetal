#!/usr/bin/python
# A Aptekmann imp of MBP1 for ENIGMA at RU
# May 2019
# Library to encode SeqRecords into features for machine learning algorithms
import random, sys, os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from inspect import signature

def mkdir(dir):
    if os.path.isdir(dir):
        return 0
    else :
        os.mkdir(dir)    

def format_id(id):
    return id.split('|')[0].replace(':','_').upper()

def histogram(data):
    plt.close()
    # fixed bin size
    bins = np.arange(0, 10, 1) # fixed bin size
    plt.xlim([0,10])
    plt.hist(data, bins=bins, alpha=0.5)
    plt.title('N of bound metals pfam')
    plt.xlabel('Bound metals')
    plt.ylabel('Count')
    plt.show()

def plot_HIS(history, lb1, attribute):
    plt.close()
    # list all data in history
    # print(history.history.keys())
    fig, ax1 = plt.subplots()
    color = 'tab:red'
    ax1.set_ylabel('Accuracy', color=color)
    ax1.set_xlabel('epoch')
    ax1.tick_params(axis='y', labelcolor=color)
    # summarize history for accuracy
    if 'acc' in history.history.keys():
        key = 'acc'
    if 'categorical_accuracy' in history.history.keys():
        key = 'categorical_accuracy'
    ax1.plot(history.history[key], color=color)
    
    color = 'tab:green'
    if 'binary_accuracy' in history.history.keys():
        key = 'binary_accuracy'
    ax1.plot(history.history[key], color=color)

    # instantiate a second axes that shares the same x-axis
    ax2 = ax1.twinx()  
    color = 'tab:blue'
    ax2.set_ylabel('Loss', color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    # summarize history for loss
    ax2.plot(history.history['loss'], color=color)
    fig.tight_layout()   
    plt.title('model accuracy and loss')
    plt.legend(['acc','loss'], loc='upper left')
    p = '/Users/aaptekmann/Desktop/MetalBindingPrediction/mbp1/ROC_plots/'      
    p = './ROC_plots/'
    mkdir(p)   
    plt.savefig(p + lb1 + '_' + attribute + '_ACC.png')        


def plot_ROC(fpr, tpr, auc, lb1, attribute):
    # just sort it on the x avxis variable (smother plot)
    data = sorted(zip(fpr,tpr), key=lambda x: x[0])
    fpr, tpr = zip(*data)
    plt.close()
#    plt.figure(1)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr, tpr, label=lb1+' (area = {:.3f})'.format(auc))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')
    p = '/Users/aaptekmann/Desktop/MetalBindingPrediction/mbp1/ROC_plots/'
    p = './ROC_plots/'
    if attribute == 'test':
        print('Not saving plot, because attribut == \'test\'')
        plt.show()
    else:    
        mkdir(p)        
        plt.savefig(p + lb1 + '_' + attribute + '_ROC.png')        
    return 0
    
def plot_PRC(precision, recall, average_precision,lb1, attribute):
    plt.close()
    data = sorted(zip(precision,recall), key=lambda x: x[1])
    precision, recall = zip(*data)
    step_kwargs = ({'step': 'post'}
                if 'step' in signature(plt.fill_between).parameters
                else {})
    plt.step(recall, precision, color='b', alpha=0.2,
            where='post')
    plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(
            average_precision))
    p = '/Users/aaptekmann/Desktop/MetalBindingPrediction/mbp1/ROC_plots/'    
    p = './ROC_plots/'
    if attribute == 'test':
        print('Not saving plot, because attribut == \'test\'')
        plt.show()
    else:    
        mkdir(p)     
        plt.savefig(p + lb1 + '_' + attribute + '_PRC.png')        
    return 0
        
def random_subset(iterator, K):
    result = []
    N = 0
    for item in iterator:
        N += 1
        if len(result) < K:
            result.append(item)
        else:
            s = int(random.random() * N)
            if s < K:
                result[s] = item
    return result

def ran_split(list_in, k ):
    k = int(k)
    random.shuffle(list_in)
    return [list_in[i::k] for i in range(k) ]

def build_Xvector(encoded):    
    X = []
    #if type(encoded[0][0]) == type('string'):
    #    encoded = remove_label(encoded)
    for x in encoded:
        X.append(x)
    X = np.asarray(X)
    return X    

def build_Yvector(encoded, value):    
    y = []
    y = [value for x in encoded ]
    y = np.asarray(y)
    return y 

def build_Xyvector(pos_encoded, neg_encoded, size_limit=None, t_size=None):
    X, y = [], []
    # Scale sample so run time is reasonable.
    print("Sample size: Neg", len(neg_encoded),"\tPos:", len(pos_encoded))
    if not size_limit :
        size_limit = max(len(pos_encoded), len(neg_encoded))    
    # Also balance dataset. 
    pos_encoded = random_subset(pos_encoded, int( min(size_limit, len(neg_encoded), len(pos_encoded) ) ) )
    neg_encoded = random_subset(neg_encoded, int( min(size_limit, len(neg_encoded), len(pos_encoded) ) ) )
    # If label in data, remove it.
    try:
        if type(pos_encoded[0][0]) == type('string') or str(type(pos_encoded[0][0])) == "<class 'numpy.str_'>":
            pos_encoded = remove_label(pos_encoded)
        if type(neg_encoded[0][0]) == type('string') or str(type(neg_encoded[0][0])) == "<class 'numpy.str_'>":
            neg_encoded = remove_label(neg_encoded) 
    except IndexError:
        print('Something went wrong when checking for labels')
        print(pos_encoded[0], '\n', neg_encoded[0])
        sys.exit(0)

    for encoded_window in pos_encoded:
        X.append(encoded_window)
        y.append(1)
    for encoded_window in neg_encoded:
        X.append(encoded_window)
        y.append(0)
        # Shuffle and split training and test sets.
    if t_size == None:
        t_size = .5
    
    if t_size == 0:
        X_train = X 
        y_train = y
        X_test = y_test = 0

    elif t_size == 1:
        X_train = y_train = 0
        X_test = X 
        y_test = y
    
    else:
        print("Shuffling and split, train and test set (t_size = %s )..." % t_size)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=t_size,
                                                           random_state=0)
    print("Done.")
    # CONVERT TO FORMAT REQUIRED by KERAS.
    X_train = np.asarray(X_train)
    X_test = np.asarray(X_test)
    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)  
    
    return X_train, y_train, X_test, y_test

####################FEATURE EXTRACTION#########################################

# Condensed AA composition
def C3_feats(coded_seq):
    C1 = float(coded_seq.count('L')) / len(coded_seq)
    C2 = float(coded_seq.count('M')) / len(coded_seq)
    C3 = float(coded_seq.count('H')) / len(coded_seq)
    return [C1, C2, C3]

# AA composition
def C20_feats(seq_record, precoded_dict):
    seq = seq_record.seq
    largo = float( len(seq)) 
    feats = [ seq.count(i) / largo  for i in precoded_dict.keys() ]
    return feats

def T3_feats(coded_seq):
    # 3 posible transitions L <-> H ; L <-> M ; M <-> H
    LH, LM, MH = 0, 0, 0
    # Reduce chars typed
    a = coded_seq
    # At max n-1 transitions
    l1 = len(coded_seq) - 1
    if l1 == 0:
        return [0.0,0.0,0.0]
    for i in range(l1):
        if a[i] == 'L':
            if a[i+1] == 'M':    # L -> M
                LM += 1
            elif a[i+1] == 'H':  # L -> H
                LH += 1
        elif a[i] == 'M':
            if a[i+1] == 'L':    # M -> L
                LM += 1
            elif a[i+1] == 'H':  # M -> H
                MH += 1
        elif a[i] == 'H':
            if a[i+1] == 'M':    # H -> M
                MH += 1
            elif a[i+1] == 'L':  # H -> L
                LH += 1
    LH, LM, MH = float(LH) / l1, float(LM) / l1, float(MH) / l1
    return [LH, LM, MH]

# Encode the distribution of an AA property
# Returns the position in sequence (expressed as a percent of sequence length)
# For the first ocurrence, 25%, 50%, 75, and last of a property
def D5_feats(coded_seq, K):
    count = coded_seq.count(K)
    if count == 0:
        return [0, 0, 0, 0, 0]
    l1 = float(len(coded_seq))
    # if K is not in ['L','M','H'] wont work
    K000 = coded_seq.index(K) / l1
    K025 = coded_seq.index(K, int(count * 0.25)) / l1
    K050 = coded_seq.index(K, int(count * 0.50)) / l1
    K075 = coded_seq.index(K, int(count * 0.75)) / l1
    K100 = coded_seq.index(K, int(count - 1)) / l1
    return [K000, K025, K050, K075, K100]

# Calls D5_feats for each class of AA (Low,Medium,High)
def D15_feats(coded_seq):
    DL = D5_feats(coded_seq, 'L')
    DM = D5_feats(coded_seq, 'M')
    DH = D5_feats(coded_seq, 'H')
    return DL + DM + DH


def CTD21_feats(seq_record, clustered_dict):
    seq = seq_record.seq
    coded_seq = [clustered_dict[i] for i in seq]
    C3 = C3_feats(coded_seq)
    T3 = T3_feats(coded_seq)
    D15 = D15_feats(coded_seq)
    return C3 + T3 + D15

def KMER_feat(seq_record, kmer_dict): 
    win_len = 5
    windows = break_into_windows(seq_record, win_len)
    Kscore = 0 
    for w in windows:
        if w in kmer_dict.keys():
            Kscore += kmer_dict[w]                    
    return [Kscore]

# This is called ONCE per record, any repetitive precalculation should be made before!    
def CTD(seq_record, precoded_dict_list, kmer_dict_list):
    # Have the record id as the first feat
    feats = [format_id(seq_record.id) ]
    # Then add the CTD feats
    for clustered_dict in precoded_dict_list:
        feats = feats + CTD21_feats(seq_record, clustered_dict)
    # Then add composition feat    
    feats = feats + C20_feats(seq_record, precoded_dict_list[0])    
    # Then add kmer feat
    for kmer_dict in kmer_dict_list:
        feats = feats + KMER_feat(seq_record, kmer_dict)    
    return feats


# Deprecate functions kept because i bet ill need them.
def break_into_windows(seq_record, win_len):
    seq = str(seq_record.seq)
    windows = [seq[i:i+win_len] for i in range(len(seq)-win_len-1)]
    return windows

def remove_label(encoded_data):
    return [ i[1:] for i in encoded_data ]

def encode(seq_string, code_d):
    encoded = [code_d[char] for char in seq_string]
    return encoded

def fasta_to_coded(seq_record_list, code_d, win_len):
    # Parse fasta to Seq_records
    print('Read %s sequences' % len(seq_record_list))
    # Break input Seq_records into windows
    seq_window_list = []
    for windows in [break_into_windows(seq_rec, win_len)
                    for seq_rec in seq_record_list]:
        for window in windows:
            seq_window_list.append(window)
    # Encode windows
    encoded = [encode(window, code_d) for window in seq_window_list]
    return encoded
