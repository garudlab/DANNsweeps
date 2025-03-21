import sys
import os

Utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '/u/project/ngarud/Garud_lab/DANN/Utils/'))
sys.path.append(Utils_path)

from GRL_multiclass_data_mergeSims_A100 import load_grl_model_lambdaScheduler, intermediate_layer_model
from ImaGene_Phased_aDNA_SLiM_mh import *

import sys


epoch = sys.argv[1]
chr = sys.argv[2]

#Load model
model_GRL=load_grl_model_lambdaScheduler('Model_odd_RowDist_201_SouilmiMD43src_batch64_lr1e-5/GRL_multiclass_THETAvary_trained') 

#Process data
#path='/u/project/ngarud/Garud_lab/aDNA/epochH/VCF.v51.0.H.chr'+str(chr)+'.vcf' #Chr2_H/VCF.v51.0.H.chr2.vcf
path='/u/project/ngarud/Garud_lab/aDNA/epochN/VCF.v40.3.EN.chr'+str(chr)+'.vcf'
path_output = 'chr'+str(chr)+'_N_pred_j10_' + str(epoch) +".txt"
window_size=201
jump = 10
file_chrom = ImaFile(nr_samples=177, VCF_file_name=path) #177

file_out = open(path_output, 'w')

#read VCF
chrom_full = file_chrom.read_VCF()
chrom_full.majorminor();
chrom_full.filter_freq(0.001); #is this too low?? maybe change this

#iterate across chromosome in sliding windows
chrom_size_SNPs = chrom_full.data[0].shape[1]
last_SNP=chrom_size_SNPs-window_size

for window_idx in range(0,last_SNP,jump):
    #get window from chromosome 
    idx = list(range(window_idx,window_idx+window_size))
    chrom_sub = chrom_full.subset_window(idx,)
    chrom_sub.sampleHaps(150) #150
    #Process and sort haps in window
    chrom_sub.sort('rows_dist') #rows_dist
    #chrom_sub.sort_centerWindow(51,'rows_freq')
    chrom_sub.convert(flip=True) 

    #re-code 
    chrom_sub.data[chrom_sub.data==0] = -1
    # substitute nan's to 0's
    chrom_sub.data[np.isnan(chrom_sub.data)] = 0
    
    #predict
    pred=model_GRL.predict(chrom_sub.data,batch_size=None)
    positions=str(int((chrom_sub.positions[0][0]+chrom_sub.positions[0][-1])/2)) + '\t'+ str(chrom_sub.positions[0][0]) + '\t' + str(chrom_sub.positions[0][-1]) # center, left , right position in window
    line_out='\t'.join(map(str, pred[0].flatten())) + '\t' + '\t'.join(map(str, pred[1].flatten()))
    print(positions + '\t'+ line_out +'\n')
    file_out.write(positions + '\t'+ line_out +'\n')
    #

file_out.close()
