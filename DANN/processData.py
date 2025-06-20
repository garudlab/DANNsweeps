### --------- load modules -------------------#
import sys
import os

Utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '/u/project/ngarud/Garud_lab/DANN/Utils/'))
sys.path.append(Utils_path)


from process_and_sort_simulations import process_slim_files
from ImaGene_Phased_aDNA_SLiM_mh import *


### --------- process simulations -------------------#
print("PROCESSING DATA")

for i in range(1,50+1):#
    print(i)
    # training data
    path= "/u/scratch/m/mharrish/ConstantNe/"
    path_out ="/u/project/ngarud/Garud_lab/DANN/aDNA/ProcessingData/"

    #Neutral
    file_in=path + "Neutral/NeutralSouilmi_THETA0.0_target0_"+str(i)+".txt"
    file_out=path_out+"ProcessedDataALL/SortByRowFreq/Neutral/SouilmiMiss2MD43_Neutral_THETA1to5_n150_w201_RowFreq_trainsrc_"+str(i)+".dat"
    gene_sim = process_slim_files(path=file_in,num_samples=150,model_name="train",\
                 classifier_name="THETA",discriminator_name='target',num_sims=8000,dimension=201,ordering='rows_freq',save_path=file_out)

    #HS
    file_in=path + "HardSweeps/HardSweepSouilmi_THETA0.01_target0_"+str(i)+".txt"
    file_out="ProcessedDataALL/SortByRowFreq/HardSweeps/Souilmi_HS_THETA1to5_n150_w201_RowFreq_trainsrc_"+str(i)+".dat"
    gene_sim = process_slim_files(path=file_in,num_samples=150,model_name="train",\
                 classifier_name="THETA",discriminator_name='target',num_sims=4000,dimension=201,ordering='rows_freq',save_path=file_out)

    #SS
    file_in=path + "SoftSweeps/SoftSweepSouilmiMiss2MD43_THETA5.0_target0_"+str(i)+".txt"
    file_out="ProcessedDataALL/SortByRowFreq/SoftSweeps/Souilmi_SS_THETA1to5_n150_w201_RowFreq_trainsrc_"+str(i)+".dat"
    gene_sim = process_slim_files(path=file_in,num_samples=150,model_name="train",\
                 classifier_name="THETA",discriminator_name='target',num_sims=4000,dimension=201,ordering='rows_freq',save_path=file_out)
    
print("TRAIN DATA --DONE")
