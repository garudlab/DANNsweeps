import numpy as np
import os.path
import sys
import random

def addMissingData(Geno_matrix,MD,sd):
     Geno_matrix=Geno_matrix.astype(object)
     alpha=(MD**2*(1-MD))/sd**2-MD
     beta=alpha/MD*(1-MD)
     n_rows=Geno_matrix.shape[0]
     for col_idx in range(Geno_matrix.shape[1]):
        f = np.random.beta(alpha,beta,1)
        #sample fraction f at each site
        nan_idx=np.random.choice(n_rows, size=int(n_rows*f), replace=False)
        Geno_matrix[nan_idx,col_idx]='N' #-1 or 'N' or NaN
     return Geno_matrix

def processSims_aDNA(inFile,outFile,windowSize):
    out_file = open(outFile,'w')

    with open(inFile, 'r') as file:
    # Read each line and print it
        line = file.readline() 
        out_file.write(line) # first line is empty
        line = file.readline() #second line if first non-empty line

         # Loop until there are no more lines
        while line:
            if line.strip(): # if no white space
                line_sep=line.split()
                if line_sep[0] == "//":
                    out_file.write(line)
                    line = file.readline()
                elif line_sep[0] == "segsites:":
                    line_out= "segsites: " + str(windowSize) + "\n"
                    out_file.write(line_out)
                    line = file.readline()
                elif line_sep[0] == "positions:" :
                    pos = line_sep[1:]
                    line = file.readline()  
                else:
                    gen_data=list(line.strip())
                    Geno_list=[]
                    while gen_data[0].isdigit(): # if line starts with 1 or 0 
                        Geno_list.append(gen_data)
                        
                        #read next line
                        line = file.readline()
                        gen_data=list(line.strip())
                        if not gen_data: # if list is empty
                            break
                    Geno_matrix= np.array(Geno_list)
                    #add missing data
                    Geno_matrix_MD=addMissingData(Geno_matrix,0.43,0.28) #0.55 0.23 0.05,0.0697, 0.43 0.28
                    
                    
                    #print(Geno_matrix_MD.shape)

                    #remove invariant sites
                    # For each column, check if it contains at least one '1' AND at least one '0'
                    has_1 = np.any(Geno_matrix_MD == '1', axis=0)
                    has_0 = np.any(Geno_matrix_MD == '0', axis=0)

                    # Only keep columns that have both '1' and '0'
                    invariant_sites = has_1 & has_0

                    # Get the indices of valid columns
                    invariant_sites_indices = np.where(invariant_sites)[0]
                    # Randomly choose 201 of them (sorted if you want)
                    invariant_sites_indices_window = sorted(random.sample(list(invariant_sites_indices), int(windowSize)))

                    # Apply the mask to keep only valid columns
                    Geno_matrix_MD_filtered = Geno_matrix_MD[:, invariant_sites_indices_window]
                    print(Geno_matrix_MD_filtered.shape)

                    #get positions
                    pos_subsample= [pos[i] for i in invariant_sites_indices_window]
                    line_out = "positions: " + " ".join(map(str, pos_subsample)) + "\n"
                    out_file.write(line_out)

                    #write to output file
                    for row in range(Geno_matrix_MD_filtered.shape[0]):
                        out_file.write("".join(Geno_matrix_MD_filtered[row,:])+"\n")

        out_file.close()



def main():

    inFile = sys.argv[1]
    outFile = sys.argv[2]
    windowSize = sys.argv[3]

    processSims_aDNA(inFile,outFile,windowSize)

#run main
if __name__ == '__main__':
    main()
