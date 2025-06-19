### --------- load modules -------------------#
import sys
import os

Utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '/u/project/ngarud/Garud_lab/DANN/Utils/'))
sys.path.append(Utils_path)


from process_and_sort_simulations import process_slim_files
from ImaGene_Phased_aDNA_SLiM_mh import *
import gc

#load data
path_data="/u/project/ngarud/Garud_lab/DANN/aDNA/ProcessingData/ProcessedDataALL/"
data_trainsrc_neu_path=path_data+'SortByRowFreq/Neutral/Souilmi_Neutral_THETA1to5_n150_w201_RowFreq_trainsrc_' #_1.dat'
data_trainsrc_hs_path=path_data+'SortByRowFreq/HardSweeps/Souilmi_HS_THETA1to5_n150_w201_RowFreq_trainsrc_'
data_trainsrc_ss_path=path_data+'SortByRowFreq/SoftSweeps/Souilmi_SS_THETA1to5_n150_w201_RowFreq_trainsrc_'

### ---- or load ImaGene object with simulations ----#
print("LOAD DATA")

gene_sim_neu_trainsrc = load_imagene(file=data_trainsrc_neu_path+str(1)+'.dat');
data_neu =gene_sim_neu_trainsrc.data.astype(np.float16) #[:,:99,:,:] #set correct dimensions 99 for CEU data

# hard sweeps
gene_sim_hs_trainsrc = load_imagene(file=data_trainsrc_hs_path+str(1)+'.dat');
data_hs =gene_sim_hs_trainsrc.data.astype(np.float16) #[:,:99,:,:] #set correct dimensions 99 for CEU data

# soft sweeps
gene_sim_ss_trainsrc = load_imagene(file=data_trainsrc_ss_path+str(1)+'.dat');
data_ss =gene_sim_ss_trainsrc.data.astype(np.float16) #[:,:99,:,:] #set correct dimensions 99 for CEU data
positions_ss = gene_sim_ss_trainsrc.positions


for i in range(2,49+1): #49,25

    # neutral
    gene_sim_neu_trainsrc_i = load_imagene(file=data_trainsrc_neu_path+str(i)+'.dat');
    gene_sim_neu_trainsrc_i.data =gene_sim_neu_trainsrc_i.data.astype(np.float16) #[:,:99,:,:] #set correct dimensions 99 for CEU data
    data_neu = np.concatenate((data_neu, gene_sim_neu_trainsrc_i.data), axis=0) 
   
    # hard sweeps
    gene_sim_hs_trainsrc_i = load_imagene(file=data_trainsrc_hs_path+str(i)+'.dat');
    gene_sim_hs_trainsrc_i.data =gene_sim_hs_trainsrc_i.data.astype(np.float16) #[:,:99,:,:] #set correct dimensions 99 for CEU data


    # soft sweeps
    gene_sim_ss_trainsrc_i = load_imagene(file=data_trainsrc_ss_path+str(i)+'.dat');
    gene_sim_ss_trainsrc_i.data =gene_sim_ss_trainsrc_i.data.astype(np.float16) #[:,:99,:,:] #set correct dimensions 99 for CEU data
    data_ss = np.concatenate((data_ss, gene_sim_ss_trainsrc_i.data), axis=0) 
    positions_ss = positions_ss +gene_sim_ss_trainsrc_i.positions


    gc.collect()

#'''
gc.collect()


data_neu[data_neu==0] = -1
data_hs[data_hs==0] = -1
data_ss[data_ss==0] = -1

# substitute nan's to 0's
data_neu[np.isnan(data_neu)] = 0
data_hs[np.isnan(data_hs)] = 0
data_ss[np.isnan(data_ss)] = 0

gc.collect()

#change data type
data_neu=data_neu.astype(np.int32)
data_hs=data_hs.astype(np.int32)
data_ss=data_ss.astype(np.int32)


#save data files
np.save('neutral_Souilmi_RowFreq_n150_w201_sims.npy', data_neu)
np.save('HS_Souilmi_RowFreq_n150_w201_sims.npy', data_hs)
np.save('SS_Souilmi_RowFreq_n150_w201_sims.npy', data_ss)

