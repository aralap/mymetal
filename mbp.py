#!/usr/bin/python
import numpy as np
from keras.models import model_from_json
import load_dicts
import iof, sys

# Standalone version just to predict 

def load_ANN(attribute):
    #encode.mkdir('./ModelPersistency/')
    fname = './ModelPersistency/ANNmodel' + str(attribute) + '.json'
    json_file = open(fname, 'r')
    # Load model architechture
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(
        './ModelPersistency/ANNmodel' +
        str(attribute) +
        ".h5")
    loaded_model.compile(loss='binary_crossentropy',
                         optimizer='rmsprop',
                         metrics=['accuracy'])
    print("Loaded model %s from disk" % attribute)
    json_file.close()
    return loaded_model

def preload(fasta):
    # load kmer counts for the ion
    ion_kmer_dict_list = load_dicts.precoded_kmer_list()
    # load dicts
    precoded_dict_list = load_dicts.precoded_dict_list()
    #fasta = '../../Fastas/pdb_seqres.txt'
    #fasta = '../../Fastas/pdb70_seqres.txt'
    # load fastas:
    a = iof.load_encode(fasta,
        precoded_dict_list, ion_kmer_dict_list, '2')#[:1000]

    # axis = 0 -> row : axis = 0 -> column
    # removing first row:
    b = np.delete(a, 0, 1)
    #After tensorflow 2.0 this became necesary
    b = b.astype('float')
    # Generating prediction as feature
    # load models and predict
    multi_model = load_ANN('Multi')
    mono_model = load_ANN('Mono')
    print('Predicting with MBP models')
    c = multi_model.predict(b)
    d = mono_model.predict(b)
    # Joining feature
    b = np.hstack((b, c, d)) 
    print('Predicting with ion models')
    metals =  ['CA', 'CO', 'CU', 'FE', 'K', 'MG', 'MN', 'NA', 'NI', 'ZN']     
    models = [load_ANN('T2'+ ion) for ion in metals ]
    p = [ model.predict(b) for model in models ]
    # a is old features.
    X_vector = np.hstack((a, c, d, p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9]))

    
    print('ID\t','Mul\t','Mono\t','CA\t', 'CO\t', 'CU\t', 'FE\t', 'K\t', 'MG\t', 'MN\t', 'NA\t', 'NI\t', 'ZN\t')
    for i in range(len(c)):
        l = [c[i], d[i], p[0][i], p[1][i], p[2][i], p[3][i], p[4][i], p[5][i], p[6][i], p[7][i], p[8][i], p[9][i]]    
        l = [round(float(i),2) for i in l]
        print(a[i][0], end='\t')
        for i in l:
            print(i, end='\t')
        print()        
      
if __name__ == "__main__":
    preload(sys.argv[1])
import code
#print('to run use: preload(fasta)')
#code.interact(local=locals())
# test  '/Users/aaptekmann/Desktop/Fastas/PDB_70_Ni_Fe.fasta'