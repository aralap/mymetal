
# By AA Aptekmann, May 3 2019, for ENIGMA project at YANALAB-RU
#
# A set of classes for storing swissprot,pdb,pfam entries
# Entry / Entry name / Protein names / Gene names
# ->/ Organism / Length / Metal binding

class blast_line():
    def __init__(self, line):
        try:
            line = line.split()
            self.bitscore = line[11]
            self.query = line[0]
            self.target = line[1]
            self.pident = line[2]
            self.length = int(line[3])
            self.mismatch = line[4]
            self.gapopen = line[5]
            self.qstart = line[6]
            self.qend = line[7] 
            self.sstart = line[8]
            self.send = line[9]
            self.evalue = float(line[10])
        except IndexError:
            #print(line, len(line))
            self.query = False    

class blast_csv():
    # default class to store blast csv result
    # Column headers:
    # qseqid sseqid pident length mismatch gapopen qstart qend sstart send evalue bitscore
    def __init__(self, record_handle):
        self.alignment = [ blast_line(line) for line in record_handle if blast_line(line).query != False ]

class meta_DB():
    
    def __init__(self):
        self.N_ENT = 0
        self.MBP = 0
        self.NMBP = 0
        self.MBC = [0,0,0,0,0,0,0,0,0,0]
        self.ENTRIES = {}

    def add_swp(self, swissprot):
        self.ENTRIES[swissprot.ENTRY] = swissprot
        self.MBC = [a + b for a,b in zip(self.MBC, swissprot.WHICH_METAL) ]
        self.N_ENT += 1
        if swissprot.METAL_BINDING == False :
            self.NMBP += 1
        else:
            self.MBP += 1

    def __add__(self, other):
        for key in other.ENTRIES.keys():
            if key not in self.ENTRIES.keys():
                self.add_swp(other.ENTRIES[key])
            elif self.ENTRIES[key] == other.ENTRIES[key]:
                   None # Entry already in DB nothing added
            else:
                print('This is awkward, you are trying to merge 2 DBs with conflicting entries:')
                print(key)
                raise Warning     
        return self

    def __repr__(self):
        t = 'Total entries:\t' + str(self.N_ENT) + '\n'
        t += '\tNon MBP entries:\t' + str(self.NMBP) + '\n'
        t += '\tMBP entries:\t' + str(self.MBP) + '\n'    
        m = ['Ca','Co','Cu','Fe','Mg','Mn','Ni','K','Na','Zn']
        for i in range(len(m)):
            t += '\t' + str(m[i]) + '\t = \t' + str(self.MBC[i]) + '\n'        
        return t 
# END OF SWISSPROT DB CLASS        

class pfam():

    def __init__(self, PDB_ID, CHAIN_ID, PdbResNumStart, 
                 PdbResNumEnd, PFAM_ACC, PFAM_Name, PFAM_desc, eValue):
        self.PDB_ID = PDB_ID
        self.CHAIN_ID = CHAIN_ID
        self.PdbResNumStart = PdbResNumStart
        self.PdbResNumEnd = PdbResNumEnd
        self.PFAM_ACC = PFAM_ACC
        self.PFAM_Name = PFAM_Name
        self.PFAM_desc = PFAM_desc 
        self.eValue = eValue

    def __repr__(self):
        return ('PDB_ID:\t' + str(self.PDB_ID) +
                '\tPFAM_ACC:\t'+ str(self.PFAM_ACC))
# END OF PFAM CLASS

class swissprot():

    def v2ls(self, vector):
        m = ['Ca','Co','Cu','Fe','Mg','Mn','Ni','K','Na','Zn']
        r = [ m[i] for i in range(len(vector)) if vector[i] ]
        return r

    def metal_interpreter(self,m_b_list):
        Ca,Co,Cu,Fe,Mg,Mn,Ni,K,Na,Zn = 0,0,0,0,0,0,0,0,0,0
        for entry in m_b_list:
            if 'Calcium' in entry:
                Ca = 1
            if 'Cobalt' in entry:
                Co = 1
            if 'Copper' in entry:
                Cu = 1
            if 'Iron' in entry:
                Fe = 1
            if 'Magnesium' in entry: 
                Mg = 1
            if 'Manganese' in entry:
                Mn = 1
            if 'Nickel' in entry:
                Ni = 1
            if 'Potassium' in entry:
                K = 1              
            if 'Sodium' in entry:
                Na = 1
            if 'Zinc' in entry:
                Zn = 1
        return [Ca,Co,Cu,Fe,Mg,Mn,Ni,K,Na,Zn]        

    def __init__(self, ENTRY, ENTRY_NAME, STATUS, PROTEIN_NAME,
                 GEN_NAME, ORGANISM, LENGTH, METAL_BINDING, PFAM):
        if METAL_BINDING != '':
            WHICH_METAL = self.metal_interpreter(METAL_BINDING.split(";"))
            METAL_BINDING = True  # METAL_BINDING.split(";")
        else:
            METAL_BINDING = False
            WHICH_METAL = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.ENTRY = ENTRY
        self.ENTRY_NAME = ENTRY_NAME
        self.STATUS = STATUS
        self.PROTEIN_NAME = PROTEIN_NAME
        self.GEN_NAME = GEN_NAME
        self.ORGANISM = ORGANISM
        self.LENGTH = LENGTH
        self.METAL_BINDING = METAL_BINDING
        self.WHICH_METAL = WHICH_METAL
        self.PFAM = PFAM.split(';')[0:-1]

    def __eq__(self, other):
        if self.ENTRY != other.ENTRY:
            return False 
        if self.ENTRY_NAME != other.ENTRY_NAME:
            return False         
        if self.STATUS != other.STATUS :
            return False
        if self.PROTEIN_NAME != other.PROTEIN_NAME:
            return False
        if self.GEN_NAME != other.GEN_NAME :
            return False
        if self.ORGANISM != other.ORGANISM :
            return False
        if self.LENGTH != other.LENGTH :
            return False
        if self.METAL_BINDING != other.METAL_BINDING :
            return False
        if self.WHICH_METAL != other.WHICH_METAL :
            return False
        if self.PFAM != other.PFAM :
            return False
        return True    

    def __repr__(self):
        return ("ENTRY:\n\t"+str(self.ENTRY) + "\nENTRY_NAME:\n\t" +
                str(self.ENTRY_NAME) + "\nPROTEIN_NAME:\n\t" +
                str(self.PROTEIN_NAME) + "\nGEN_NAME:\n\t" +
                str(self.GEN_NAME) + "\nORGANISM:\n\t" +
                str(self.ORGANISM) + "\nLENGTH:\n\t" +
                str(self.LENGTH) + "\nMETAL_BINDING:\n\t" +
                str(self.METAL_BINDING)+"\nWHICH_METAL:\n\t" +
                str(self.v2ls(self.WHICH_METAL))+"\nPFAM:\n\t" +
                str(self.PFAM)+"\n")
# END OF SWISSPROT CLASS
