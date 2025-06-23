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

  This script will generate n+2 files where n= number of training epochs:
  - The file _GRL_multiclass_model.json_ contains the model architechture
  -  _n=30_ files (_GRL_multiclass.1.weights.h5,GRL_multiclass.2.weights.h5,...GRL_multiclass.n.weights.h5_) contain the weights after each epoch of training.
  -  The file _training_multiclass_results.txt_ with the target and source accuracies and losses for each epoch of training (30 epochs in this script, but this can be adjusted in the code).

# 3) Choosing the model weights

To choose the best training epoch weights for our model, we would usually choose the epoch with lowest validation accuracy in the target data. This is only possible when we know the labels for the target data, which is only the case when we're benchmarking with simulations. But in the real application, our target is unlabeled and hence, we don't have a validation set. In this scenario, we test on labeled data from the source domain. For the weights from each training epoch, we compute the area under the precision-recall curve (AUPRC) in a one-vs-rest approach (HS vs other classes, SS vs other classes and Neutral vs other classes). We then take the average of these three AUPRCs for each training epoch and choose the weights of the epoch with highest average AUPRC.

The script to compute the AUPRC per epoch is _Test_wSimulations_auprc.py_ as input it requires the following:
-  **number of training epoch**
- **name of the model:** _model_name='GRL_multiclass'_
- **file paths to test data sets for each class:**
  - neutral_path ="/u/project/ngarud/Garud_lab/DANN/aDNA/ProcessingData/ProcessedDataALL/SortByRowFreq/Neutral/ConstantNeMD43_Neutral_THETA1to5_n150_w201_RowFreq_trainsrc_50.dat"
  - HS_path = "/u/project/ngarud/Garud_lab/DANN/aDNA/ProcessingData/ProcessedDataALL/SortByRowFreq/HardSweeps/ConstantNeMD43_HS_THETA1to5_n150_w201_RowFreq_trainsrc_50.dat"
  - SS_path = "/u/project/ngarud/Garud_lab/DANN/aDNA/ProcessingData/ProcessedDataALL/SortByRowFreq/SoftSweeps/ConstantNeMD43_SS_THETA1to5_n150_w201_RowFreq_trainsrc_50.dat"

We can run the code as define environment and variables in _qsub_TestAUPRC_ and submit the job to the cluster with

```bash
  qsub qsub_TestAUPRC
```

Runnign this code will output a file _testing_auprc_results.txt_ with 30 rows (one per epoch), with columns: _epoch_, _AUPRC_neutral_, _AUPRC_HS_,_AUPRC_SS_ and _AUPRC_Sweep_. Where for  _AUPRC_Sweep_ we merge toghether the hard and soft sweep categories and compute the binary AUPRC (Sweep vs neutral), the other three are the one vs. rest AUPRCs as described above. 

A sample of the first 5 lines of the output file is given below:

```bash
  1,0.88,0.84,0.61,0.94
  2,0.90,0.86,0.68,0.95
  3,0.90,0.87,0.70,0.95
  4,0.90,0.88,0.72,0.94
  5,0.90,0.88,0.72,0.95
```

