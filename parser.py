#!/usr/bin/python3
from Bio import SeqIO
import random
from progress.bar import Bar
import encode  # AA Lib 
import test # AA Lib
from classes import pfam, swissprot, meta_DB
# parsing functions for metadata and mixed fasta:

def parse_fasta(fasta_filename):
    seq_record_list = [rec for rec in SeqIO.parse(fasta_filename, "fasta")]
    return seq_record_list

def separe_neg_pos(filename, meta, key, value):
    # Parse fasta to Seq_records
    seq_record_list = parse_fasta(filename)
    # Separate data by key and value
    positive_data, negative_data = [], []
    for record in seq_record_list:
        if getattr(meta[record.id.split('|')[1]], key) == value:
            positive_data.append(record)
        else:
            negative_data.append(record)
    print("Negative sample size:", len(negative_data))
    print("Positive sample size:", len(positive_data))
    return positive_data, negative_data

def sep_by_pfam(filename, metadata, test_set_size, key, value):
    # Parse fasta to Seq_records
    seq_record_list = parse_fasta(filename)
    main_set, test_set = [], []
    print('Performing homology separation, separating %s PFAM families' % int(test_set_size * len(seq_record_list)) )

    # Which PFAMS are present?
    pfam_families = [ metadata[i].PFAM for i in metadata.keys()  if getattr(meta.ENTRIES[i], key) == value ]
    pfam_families = set([i for x in pfam_families for i in x ])

    # Chooose randomly a group of PFAMs to exclude
    choose = int(len(pfam_families) * test_set_size )
    excluded_pfams = set(random.sample(pfam_families, choose ) )
   
    # Iterate data to separate : 
    bar = Bar('Building record list.', max=len(seq_record_list))
    for record in seq_record_list:
        bar.next()
        uniprot_code = record.id.split('|')[1]
        if  set(metadata[uniprot_code].PFAM) & excluded_pfams != set():
            test_set.append(record)
        else:
            main_set.append(record)  
    bar.finish()
    print(len(test_set),len(main_set))
    return main_set, test_set, pfam_families

def is_protein(record):
    ProtAlph = 'DEFHIKLMNPQRSVWY' #removed ACGT
    for i in ProtAlph:
        if i in record.seq:
            return True
    return False        

def getid(record,flag):
    if flag == 0:
        # flag == 0 means we are reading from a File
        if '|PDBID|CHAIN|SEQUENCE' in record.id:
            # This is the tipical record id for PDB fastas
            t = record.id.split('|')[0].split(':')
            pdb_id = t[0]
            chain_id  = t[1]
            r_id = pdb_id + '_' + chain_id 

            return r_id  
        else:
            #print(record.id.upper())
            # Other fasta formats
            return record.id.upper()
    elif flag == 1:
        r_id = record[0].split('_')
        pdb_id = r_id[0]
        assert len(r_id) == 2, 'malformed PDB id. (no chain id or _ absent)'
        assert len(pdb_id) == 4, 'malformed PDB id. (id len != 4) '
        #print(record[0].upper())
        return record[0].upper()
    else:
        raise Warning

def sep_by_cdhit(file_or_vectorlist, pdb_clusters, test_set_size_or_k):
    print('Separating by homology', end='')    
    # CHECK IF SORTING SEQ RECORDS OR VECTORS
    if type(file_or_vectorlist) == type('string'):
        seq_record_list = parse_fasta(file_or_vectorlist)
        flag = 0
    elif type(file_or_vectorlist) == type(['string',1.0,2.0,3.0]):
        seq_record_list = file_or_vectorlist
        flag = 1
    elif str(type(file_or_vectorlist)) == '<class \'numpy.ndarray\'>':
        seq_record_list = file_or_vectorlist
        flag = 1
    else:
        print(type(file_or_vectorlist))
        print('encode.py> sep_by_cdhit: only takes a list of vectors or a fasta file as input')
        raise Warning
    cdhit_clust = list(set(pdb_clusters.values()))
    train_records = [] ; test_records = []

    if test_set_size_or_k <= 1: 
        print(' into 2 groups (%s/%s)' % (1-test_set_size_or_k, test_set_size_or_k))    
        # Chooose randomly a group of CDHIT clusters to exclude
        choose = int(max(pdb_clusters.values()) * test_set_size_or_k)
        excluded_clusters = random.sample( cdhit_clust, choose ) 
        # init out lists    
        for record in seq_record_list:
            if pdb_clusters[getid(record, flag)] in excluded_clusters:
                test_records.append(record)
            else: 
                train_records.append(record)
        return train_records, test_records

    elif test_set_size_or_k > 1:
        print(' into %s groups.' %  test_set_size_or_k)    
        cluster_groups = encode.ran_split(cdhit_clust, test_set_size_or_k)
        #print(cluster_groups)
        for k in cluster_groups:
            temp = []
            for record in seq_record_list:
                pdb_id = getid(record,flag)
                #print(pdb_id, pdb_id in pdb_clusters.keys(), pdb_clusters[getid(record,flag)] in k ) 
                if pdb_id in pdb_clusters.keys():
                    if pdb_clusters[getid(record,flag)] in k:
                        temp.append(record)  
            #seqs2short = [ getid(record,flag) for record in seq_record_list if getid(record,flag) not in pdb_clusters.keys()  ]
            #print(len(temp), len(seqs2short))
            #print(seqs2short)
            train_records.append(temp)
        c = 0
        for i in train_records:
            c+=1
            #print('Cluster group %s ids:' % c )
            #print([getid(record,flag) for record in i])    
        return train_records

    else:
        raise Warning

def sep_by_chain(filename, metadata, pdb_chain):
    print('Separating by chain.')
    # Parse fasta to Seq_records
    seq_record_list = parse_fasta(filename)
    pos_chain, neg_chain, neg_pdb = [], [], []
    no_chain, nucleic = [], []
    mono = []
    mult = []
    unknowns = []
    # Iterate data to separate : 
    bar = Bar('Building record lists.', max=len(seq_record_list))
    for record in seq_record_list:
        bar.next()
        if is_protein(record):
            # Parse id
            id = record.id.upper().split('_')
            pdb_id = id[0]
            chain = id[1]
            # Save id, if repeated is a multimer 
            if pdb_id in mono:
                mult.append(pdb_id)
                mono.remove(pdb_id)
            elif pdb_id not in mult:
                mono.append(pdb_id)

            # Assign to one category
            if pdb_id in pdb_chain.keys():
                if pdb_chain[pdb_id] == []:
                    # No chain in pdb is in vecinity of M
                    neg_pdb.append(record)        

                elif chain in pdb_chain[pdb_id]:
                    # this chain binds a metal
                    pos_chain.append(record)

                else :
                    # this chain doesnt
                    neg_chain.append(record)
            else:
                # id is not present 
                no_chain.append(record)
                unknowns.append(str(id)+'\t'+str(chain))
        else:
            # record is a nucleic acid
            nucleic.append(record)    
    bar.finish()

    mult_pos = [record for record in pos_chain if record.id.upper().split('_')[0] in mult]
    mult_neg = [record for record in neg_chain+neg_pdb if record.id.upper().split('_')[0] in mult]
    mono_pos = [record for record in pos_chain if record.id.upper().split('_')[0] in mono]
    mono_neg = [record for record in neg_chain+neg_pdb if record.id.upper().split('_')[0] in mono]

    # no_chain_set = set([i.id.upper().split('_')[0] for i in no_chain]) -> LARGE structures without a PDB file       
    # print('\n'.join(no_chain_set))    

    print('Recs:', len(seq_record_list), '\tChain-:', len(neg_chain), '\tChain+:', len(pos_chain), '\tPDB-:', len(neg_pdb), '\tUnKn:', len(no_chain), '\tNucl:', len(nucleic))          
    print('Mult:', len(mult), '\tMult-:', len(mult_neg), '\tMult+', len(mult_pos))
    print('Mono:', len(mono), '\tMono-:', len(mono_neg), '\tMono+:', len(mono_pos)) 
    #iof.save_out(unknowns, 'no_3dinfo.txt') 
    return pos_chain, neg_chain, neg_pdb, mult_pos, mult_neg, mono_pos, mono_neg 

# PARSE METADATA FROM UNIPROTS 
def parse_metadata(metadata_tabformat):
    print('Loading Uniprot medatada on MBP.')
    metadata = meta_DB()
    handle = open(metadata_tabformat, 'r')
    for line in handle:
        if line.startswith("Entry"):
            break
        else:
            print(line)
            print('Reached EOF without finding header starting with \'Entry\'')
            handle.close()
    for line in handle:
        line = line.split('\t')
        ENTRY = line[0]
        ENTRY_NAME = line[1]
        STATUS = line[2]
        PROTEIN_NAME = line[3]
        GEN_NAME = line[4]
        ORGANISM = line[5]
        LENGTH = line[6]
        METAL_BINDING = line[7]
        PFAM = line[8]
        #Swissprot codes are unique
        swis_obj = swissprot(ENTRY, ENTRY_NAME, STATUS, PROTEIN_NAME,
                                    GEN_NAME, ORGANISM, LENGTH, METAL_BINDING, PFAM)
        metadata.add_swp(swis_obj)    
    handle.close()
    return metadata
###############################################################################
# Needed for PDB PFAM mapping, uses pfam class.
def parse_pfam(pfam_txt):
    print('Loading PDB to PFAM mapping.')
    pdbid_pfam = []
    handle = open(pfam_txt,'r')
    for line in handle:
        line = line.split('\t')
        PDB_ID = line[0]
        CHAIN_ID = line[1]
        PdbResNumStart = line[2]
        PdbResNumEnd = line[3]
        PFAM_ACC = line[4]
        PFAM_Name = line[5]
        PFAM_desc = line[6] 
        eValue = line[7]
        # More than one record could exist for each PDB code
        pdbid_pfam.append(pfam(PDB_ID, CHAIN_ID, PdbResNumStart, 
                 PdbResNumEnd, PFAM_ACC, PFAM_Name, PFAM_desc, eValue) )

    return pdbid_pfam

def parse_pdb_ligand(pdb_ligand_csv_file):
    print('Loading PDB ligand data.')
    pdbid_ligand = {}
    handle = open(pdb_ligand_csv_file,'r')
    for line in handle:
        line = line.rstrip('\n').split('\t')
        PDB_ID = line[0]
        LIGANDS = line[1:]
        pdbid_ligand[PDB_ID] = LIGANDS
    return pdbid_ligand


def load_pdb_swissprot_mapping(pdb_swissprot_map_txt):
    print('Loading PDB to Uniprot mapping.')
    handle = open(pdb_swissprot_map_txt,'r')
    pdb_sws = {}
    pdbchain_sws = {}
    for line in handle:
        line = line.split()
        try:
            PDBCODE = str(line[0])
            CHAIN = line[1]
            SWISSPROT = line[2]
            if SWISSPROT != '?':
                pdbchain_sws[PDBCODE.upper() + '_' + CHAIN] = SWISSPROT
                pdb_sws[PDBCODE.upper() ] = SWISSPROT
            else:
                pdbchain_sws[PDBCODE.upper() + '_' + CHAIN] = None
                pdb_sws[PDBCODE.upper() ] = None
        except IndexError:
            #print(line)    
            pdbchain_sws[PDBCODE.upper() + '_' + CHAIN] = None
            pdb_sws[PDBCODE.upper() ] = None

    return pdb_sws # , pdbchain_sws   

# Parse a CSV with info on which chains are within treshold of a metal atom
# Line has this format:
# PDB_ID\tPOS_CHAIN_1\tPOS_CHAIN2 
def parse_pdb_chain(pdb_chain_map_csv):
    print('Loading PDB chain mapping %s .' % str(pdb_chain_map_csv.split('/')[-1:]), end='... ' )
    handle = open(pdb_chain_map_csv,'r')
    pdb_chain = {}
    for line in handle:
        line = line.split()
        PDBCODE = str(line[0])
        try:
            CHAIN = line[1:]
        except IndexError:
            CHAIN = []
        pdb_chain[PDBCODE] = CHAIN     
    print('Done.')       
    return pdb_chain

def parse_cdhit(cdhit_clusters):    
    print('Loading CDHIT cluster chain mapping.')
    handle = open(cdhit_clusters,'r')
    pdb_clusters = {}
    for line in handle:
        if line.startswith('>Cluster'):
            cluster = int(line.split()[1])
        else :
            line = line.split(', >')[1].split('...')[0]
            PDB_ID = line.split('_')[0].upper()
            CHAIN_ID = line.split('_')[1].upper()
            pdb_clusters[PDB_ID+'_'+CHAIN_ID] = cluster       
    return pdb_clusters  

if __name__ == "__main__":  
    # LOAD DATA  
    # Just to save chars on repeating this folder
    path = '/Users/aaptekmann/Desktop/'
    # Define the source of swissprot seqs
    in_file = path+'/Fastas/all_uniprot_reviewed.fasta'
    # Mapping from PDB to PFAM
    pdb_pfam = parse_pfam(path+'/Metadata/pdb_pfam_mapping.txt')
    # Mapping from PDB to UNIPROT
    #pdb_uniprot = load_pdb_swissprot_mapping(path+'/Metadata/pdb_swissprot_mapping.txt')
    pdb_uniprot = load_pdb_swissprot_mapping(path+'/Metadata/pdb_chain_uniprot.lst')
    uniprot_pdb =  dict([reversed(i) for i in pdb_uniprot.items()])     
    #Load homology clusters:
    pdb_clusters = parse_cdhit(path + '/Metadata/pdb70_seqres.txt.clstr')
    #Metadata with PFAM
    meta = parse_metadata(path + '/Metadata/all_uniprot_wpfam.tab')
    meta2 = parse_metadata(path + '/Metadata/all_uniprot_wpfam_extra.tab')
    meta = meta + meta2
    # PDB ligand metadata
    pdbid_ligand = parse_pdb_ligand(path + '/Metadata/pdb_ligand.csv')
    # PDB chain-metal metadata
    pdb_chain  = parse_pdb_chain(path + '/PDB_DB/PDB_chain_3.0A.csv') 
######################################
# /  \.-"""-./  \
# \    -   -    / You got here so fast,
#  |   o   o   |  you broke the  
#  \  .-'''-.  /  "sounBEARrier" 
#   '-\__Y__/-'
#      `---`
# Perform clustering by some criteria?

#    test.discrepancy(pdbid_ligand,pdb_uniprot,meta)
    test_set_size = .2
#    train, test  = sep_by_cdhit(path + '/Fastas/pdb_seqres.txt', pdb_clusters, test_set_size )
#    pos_chain, neg_chain, neg_pdb, mult_pos, mult_neg, mono_pos, mono_neg  = sep_by_chain(path + '/Fastas/pdb_seqres.txt', meta, pdb_chain )
#    SeqIO.write(mult_neg, path + '/Fastas/pdb_mult_neg_chain.fasta', "fasta")
    # Separated by chain only
#    SeqIO.write(pos_chain, path + '/Fastas/pdb_pos_chain.fasta', "fasta")
#    SeqIO.write(neg_chain, path + '/Fastas/pdb_neg_chain.fasta', "fasta")
#    SeqIO.write(neg_pdb, path + '/Fastas/pdb_neg_pdb.fasta', "fasta")
    # Separated by multimery (Number of chains) 
#    SeqIO.write(mult_neg, path + '/Fastas/pdb_mult_neg_chain.fasta', "fasta")
#    SeqIO.write(mult_pos, path + '/Fastas/pdb_mult_pos_chain.fasta', "fasta")
#    SeqIO.write(mono_neg, path + '/Fastas/pdb_mono_neg_chain.fasta', "fasta")
#    SeqIO.write(mono_pos, path + '/Fastas/pdb_mono_pos_chain.fasta', "fasta")

#    main_set, test_set, pfams_present_n = sep_by_pfam(path + '/Fastas/temp_positives.fasta', meta, test_set_size, 'METAL_BINDING', True)
#    SeqIO.write(main_set, path + '/Fastas/temp_positives_main.fasta', "fasta")
#    SeqIO.write(test_set, path + '/Fastas/temp_positives_test.fasta', "fasta")

#    main_set, test_set, pfams_present_p = sep_by_pfam(path + '/Fastas/temp_negatives.fasta', meta, test_set_size, 'METAL_BINDING', False)
#    SeqIO.write(main_set, path + '/Fastas/temp_negatives_main.fasta', "fasta")
#    SeqIO.write(test_set, path + '/Fastas/temp_negatives_test.fasta', "fasta")

#    print('There are:\n\t %s PFAMs in positive dataset and,\n\t %s PFAMs in negative dataset' % (len(pfams_present_p), len(pfams_present_n) ) )
#    print('And %s PFAMs are common to both sets' % len(set(pfams_present_n) & set(pfams_present_p) ) )    
