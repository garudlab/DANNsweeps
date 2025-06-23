import sys
import os

Utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '/u/project/ngarud/Garud_lab/DANN/Utils/'))
sys.path.append(Utils_path)

from GRL_multiclass_data_mergeSims_A100 import load_grl_model_lambdaScheduler, intermediate_layer_model
from ImaGene_Phased_aDNA_SLiM_mh import * 

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
#from sklearn.manifold import TSNE
from matplotlib.colors import ListedColormap
from sklearn.metrics import precision_recall_curve, auc, confusion_matrix
import sys

def compute_precision_recall(y_true,y_pred,n_classes):   
   precision = dict()
   recall = dict()
   pr_auc = dict()
   for i in range(n_classes):
      precision[i], recall[i], _ = precision_recall_curve(y_true[:, i], y_pred[:, i])
      pr_auc[i] = auc(recall[i], precision[i])
   
   return precision,recall,pr_auc


def precision_recall(class_labels,y_grl_pred,model_GRL,epoch):   #gene_sim_src_test

    #get number of classes in data
    n_classes_src = 3+1

    #combine hard + soft
    combine_src_sweeps = np.logical_or(class_labels[:, 1],class_labels[:, 2]).astype(int).reshape(-1, 1)
    y_true_src=np.hstack([class_labels, combine_src_sweeps])

    #probabilities
    combine_grl_sweeps_pred = np.maximum(y_grl_pred[0][:, 1], y_grl_pred[0][:, 2]).reshape(-1, 1)

    y_pred_grl_combined = np.hstack([y_grl_pred[0], combine_grl_sweeps_pred])
   

    #Calcultate PR curves
    precision_grl,recall_grl,pr_auc_grl=compute_precision_recall(y_true_src,y_pred_grl_combined,n_classes_src)
    

    #output to file
    # Save to a text file
    with open('testing_auprc_results.txt', 'a') as f:
        f.write(f"{epoch},{pr_auc_grl[0]:.4f},{pr_auc_grl[1]:.4f},{pr_auc_grl[2]:.4f},{pr_auc_grl[3]:.4f}\n")

    '''
    #PLOT PRC
    class_labels = ["Neutral", "Hard sweep", "Soft sweep","Sweeps"]
    colors_GRL = ["#BAB0AC", "#FF9D9A", "#A0CBE8","#D4A6C8"]
    path ="auprc_"+str(epoch)+".png"
    plt.figure(figsize=(8, 6))
    for i in range(n_classes_src):
      plt.plot(recall_grl[i], precision_grl[i], color=colors_GRL[i],linestyle='-', lw=2,  label=f'{class_labels[i]} AUC-PRC = {pr_auc_grl[i]:.2f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
    plt.savefig(path)
    plt.close()
    '''

# -----------------------------------------------------------------------------------------------------------------------------------

## main

### read epoch and file names
epoch = sys.argv[1]
model_name = str(sys.argv[2])
data_trainsrc_neu_path = str(sys.argv[3])
data_trainsrc_hs_path = str(sys.argv[4])
data_trainsrc_ss_path = str(sys.argv[5])

##### Load test data ##########
gene_sim_neu_trainsrc = load_imagene(file=data_trainsrc_neu_path); 
data_neu =gene_sim_neu_trainsrc.data.astype(np.float16) 
positions_neu = gene_sim_neu_trainsrc.positions
# hard sweeps
gene_sim_hs_trainsrc = load_imagene(file=data_trainsrc_hs_path);
data_hs =gene_sim_hs_trainsrc.data.astype(np.float16) 
positions_hs = gene_sim_hs_trainsrc.positions
# soft sweeps
gene_sim_ss_trainsrc = load_imagene(file=data_trainsrc_ss_path);
data_ss =gene_sim_ss_trainsrc.data.astype(np.float16)
positions_ss = gene_sim_ss_trainsrc.positions


gene_sim_neu = ImaGene(data=data_neu, positions=positions_neu)
gene_sim_hs = ImaGene(data=data_hs, positions=positions_hs)
gene_sim_ss = ImaGene(data=data_ss, positions=positions_ss)

gene_sim_neu.data[gene_sim_neu.data==0] = -1
gene_sim_hs.data[gene_sim_hs.data==0] = -1
gene_sim_ss.data[gene_sim_ss.data==0] = -1

# substitute nan's to 0's
gene_sim_neu.data[np.isnan(gene_sim_neu.data)] = 0
gene_sim_hs.data[np.isnan(gene_sim_hs.data)] = 0
gene_sim_ss.data[np.isnan(gene_sim_ss.data)] = 0


gene_sim_src_test_data = np.concatenate((gene_sim_neu.data, gene_sim_hs.data,gene_sim_ss.data), axis=0) 

#load models
model_GRL=load_grl_model_lambdaScheduler(model_name+'_trained')

#Precision Recall
y_pred_grl= model_GRL.predict(gene_sim_src_test_data)

#labels
neutral_label = np.array([1., 0., 0.])
HS_label = np.array([0., 1., 0.])
SS_label = np.array([0., 0., 1.])

print(len(gene_sim_neu.data))

neutral_class_vstack=np.vstack([neutral_label]*len(gene_sim_neu.data))
hs_class_vstack=np.vstack([HS_label]*len(gene_sim_hs.data))
ss_class_vstack = np.vstack([SS_label]*len(gene_sim_ss.data))

class_labels = np.concatenate((neutral_class_vstack,hs_class_vstack,ss_class_vstack))
      
precision_recall(class_labels,y_pred_grl,model_GRL,epoch)  


