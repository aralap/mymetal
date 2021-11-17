#!/usr/bin/python
# A Aptekmann imp of MBP1 for ENIGMA at RU
# May 2019
# Library to prepare dictionaries for machine learning algorithms

import jenks  # AA lib


class amino():
    # L1   = one letter code
    # H1   = hidrophobicity
    # H2   = hidrophicility
    # Hb   = n of potential h bonds
    # VOL  = side_chain_vol
    # P1   = polarity
    # p2   = polarizability
    # SASA = solvent accesible surface
    # NCI  = net charge index of side chains
    # MASS = average mass of amino acid
    # doi:10.1371/journal.pone.0147467.t002
    # Added:
    # X (Undetermined AA),
    # U (selenocisteine),
    # Z (either glutamic acid or glutamine, so an average of both.),
    # B (Aspartic acid or Asparagine),
    # J (Leucine or Isoleucine),
    # O (Pyrrolisine, coded with Lysine chem data, except for MASS)
    def __init__(self, L1, H1, H2, Hb, VOL, P1, P2, SASA, NCI, MASS):
        self.L1 = L1
        self.H1 = H1
        self.H2 = H2
        self.Hb = Hb
        self.VOL = VOL
        self.P1 = P1
        self.P2 = P2
        self.SASA = SASA
        self.NCI = NCI
        self.MASS = MASS


def load_aa():
    aa_dict = {}
    handle = open("aa.csv", "r")
    for line in handle:
        if line.startswith("Amino"):
            break
    for line in handle:
        temp = line.rstrip("\n").split(",")
        temp = tuple(temp)
        new_aa = amino(*temp)  # '*' expands tuple into argument
        aa_dict[new_aa.L1] = new_aa
    handle.close()
    return aa_dict

def precoded_kmer_list():
    precoded_kmer_list = []
    metals = ['CA', 'CO', 'CU', 'FE', 'K', 'MG', 'MN', 'NA', 'NI', 'ZN', '']
    for x in metals:
        precoded_kmer_list.append( load_kmer(x) )
    return precoded_kmer_list

def load_kmer(ion=''):
    print('Loading  %s Kmer Dictionary.' % ion)
    kmer_dict = {}
    metals = ['CA', 'CO', 'CU', 'FE', 'K', 'MG', 'MN', 'NA', 'NI', 'ZN']
    s = './kmer_counts/'
    assert ion in metals or ion == '', ' Error loading kmer dict'  

    handle = open(s + "5mer_5Acounts"+ion+".txt", "r") 
    for line in handle:
        temp = line.rstrip("\n").split()
        if len(temp) > 1 :
            kmer = str(temp[0])
            count = int(temp[1])
            kmer_dict[kmer] = count
    handle.close()
    return kmer_dict 

# A method to translate a analog value AA property dictionary
# to 3 discrete categories (See YZ Chen 2006 on notes. )
def cluster_k1(dictAA):
    dataList = sorted([float(i) for i in dictAA.values()])
    numClass = 3
    clustered_dict = {}
    # Clusterize by jenks method into 3 groups
    kclass = jenks.getJenksBreaks(dataList, numClass)[1::]

    # Low, Mid, High
    kclassD = {kclass[0]: 'L', kclass[1]: 'M', kclass[2]: 'H'}

    for key in dictAA:
        # Check value of the property for an AA
        val = float(dictAA[key])
        # Find nearest cluster
        class_val = min(kclass, key=lambda x: abs(x-val))
        # Map to the nearest val
        clustered_dict[key] = kclassD[class_val]
    return clustered_dict


def precoded_dict_list():
    # Load database of aminoacid properties
    aa_db = load_aa()
    precoded_dict_list = []
    for attr in ['H1', 'H2', 'Hb', 'VOL', 'P1', 'P2', 'SASA', 'NCI', 'MASS']:
        print('Testing attribute:', attr)
        code_d = {k: getattr(aa_db[k], attr) for k in aa_db.keys()}
        clustered_dict = cluster_k1(code_d)
        precoded_dict_list.append(clustered_dict)
    return precoded_dict_list
