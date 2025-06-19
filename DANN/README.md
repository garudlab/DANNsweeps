# 1) Input data for training
The DANN takes haplotype matrices of "images" as input, where each row represents a pseudo-haplotype from a sample and 
each column the position of each variant site in the sample. These haplotype matrices have dimensions n x L, where we set n=150 
pseudo-haplotypes and L=201 SNPs in a given window. 
In these images, the color of each pixel represents the occurrence of the major or minor allele, where 
we transformed the alleles into binary values such that the major allele was coded as -1 and minor allele as 1. Missing data was coded as 0. 

We sort the haplotype matrices that we use as input for better performance of the model. We try three different sorting approaches:

1) Sort by frequency of most to least common haplotype
2) Sort by distance to most frequent haplotype
3) Sort a central window by frequency of most to least common haplotype

In **processData.py** we must define the path to the simulated data and the path for our output. We must also specify the number of samples in per simulation (or rows of haplotype matrix, num_samples=150), the total number of simulations (num_sims=4000) per file, the number of SNPs (dimension=201) and the sorting approach we are using (ordering='rows_freq'). Once everything is defined we run 

```bash
qsub qsub_processData #set correct virtual environment
```
For each file that we process through **processData.py**  we get a .dat object generated using a modified version of ImaGene (Torada et al, 2019), **ImaGene_Phased_aDNA_SLiM_mh.py**. Next, we merge the sorted matrices from all our sorted .dat objects and output an numpy object that will serve as input to our model.

```bash
qsub qsub_DataForTraining #set correct virtual environment
```
After sorting and merging all simulations we shoudl end up with three *.npy files one for neutral simulations, one for hard sweeps and one for soft sweeps. This is the input data that we will use as the source domain to train the classifier.

To generate the aDNA target data used for training the discriminator follow the documentation in ../DataProcessing/ 

# 2) Training DANN

Once the simulations have been generated and processed and the real aDNA has been processed we are ready to train the DANN.
To train the model we submit the job **qsub_TrainModel**, which is running the script **main_train.py**.

To run this script we have to define the following within the **qsub_TrainModel** script:
* **Model name.** This is the model name we use for the optput (e.g _model_name='GRL_multiclass'_).

  During training we will generate n+1 files where n= number of training epochs. The file _GRL_multiclass_model.json_ contains the model architechture and the remaining n files (_GRL_multiclass.1.weights.h5,GRL_multiclass.2.weights.h5,...GRL_multiclass.n.weights.h5_) contain the weights after each epoch of training.
* **path to training data.** We define four separate paths to our training data:
  - mmap_neutral = '/u/project/ngarud/Garud_lab/DANN/aDNA/ProcessingData/NPYprocessedfilesALLSnps/neutral_ConstantNeMD43_RowFreq_n150_w201_sims.npy' # neutral simulations
  - mmap_HS = '/u/project/ngarud/Garud_lab/DANN/aDNA/ProcessingData/NPYprocessedfilesALLSnps/HS_ConstantNeMD43_RowFreq_n150_w201_sims.npy' # Hard sweep processed simulations
  - mmap_SS = '/u/project/ngarud/Garud_lab/DANN/aDNA/ProcessingData/NPYprocessedfilesALLSnps/SS_ConstantNeMD43_RowFreq_n150_w201_sims.npy'# Soft sweeps processed simulations
  - mmap_target = '/u/project/ngarud/Garud_lab/aDNA/TrainingData/target_N-H_ALL_RowFreq_n150_w201.npy' # real processed aDNA data

To submit the job run: 
```bash
qsub qsub_TrainModel
```
Or run directly using

```bash
python main_train.py model_name mmap_neutral mmap_HS mmap_SS mmap_target
```

  within the _main_train.py_ script you can adjust the batch size (set as 64 by default)

