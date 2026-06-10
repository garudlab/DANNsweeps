### --------- load modules -------------------#
import sys
import os

Utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '/u/project/ngarud/Garud_lab/DANN/Utils/'))
sys.path.append(Utils_path)

from ImaGene_Phased_aDNA_SLiM_mh import *

### --------- functions -------------------#

def process_slim_files(path,num_samples,model_name, classifier_name,discriminator_name, num_sims,dimension,ordering,save_path):
    """
    Process SLiM MS files 

    Keyword Arguments:
        path (string) -- path to simulation files
        num_samples (int) -- number of indivuduals in sample
        model_name (sting) -- name of Model
        parameter (string) -- parameter we want to predict i.e. THETA
        num_sims (int) -- number of simulations per combination of parameters
        dimension (int) -- number of SNPs in window

    Return:
        imagene object
    """
    #create ImaGene object
    file_sim = ImaFile(simulations_folder=path, nr_samples=num_samples, model_name=model_name);

    #read simulations
    gene_sim_sweep = file_sim.read_simulations(classifier_name=classifier_name,discriminator_name=discriminator_name, max_nrepl=num_sims)

    # -- re-arrange data
    # switch to major/minor allele polarisation
    gene_sim_sweep.majorminor()
    gene_sim_sweep.sampleHaps(150)

    #get correct dimensions
    gene_sim_sweep.crop(dimension);

    #sort rows by genetic distance
    if ordering == 'rows_dist':
        gene_sim_sweep.sort('rows_dist') # you can also sort by: 'rows_dist', 'rows_freq' or 'cols_freq' or a combination of these
    elif ordering == 'rows_freq':
        gene_sim_sweep.sort('rows_freq')
    elif ordering == 'center_window_dist':
        gene_sim_sweep.sort_centerWindow(51,'rows_dist')
    elif ordering == 'center_window_freq':
        gene_sim_sweep.sort_centerWindow(51,'rows_freq')
    else:
        print('Select a valid ordering.')
        return 1

    gene_sim_sweep.convert(flip=True) 

    # define classes
    gene_sim_sweep.classes_classifier = np.array([0,0.01,5.0]) #[0,0.01,10],[0,0.01] [0,10.0]
    gene_sim_sweep.classes_discriminator = np.array([0.0,1.0])


    #convert targets to categorical data
    gene_sim_sweep.targets_classifier = to_categorical(gene_sim_sweep.targets_classifier) #to_categorical

    #and convert  domain to binary data 
    gene_sim_sweep.targets_discriminator = to_binary(gene_sim_sweep.targets_discriminator )

    gene_sim_sweep.save(file=save_path)

    return gene_sim_sweep
