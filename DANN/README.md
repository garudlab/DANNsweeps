# 1) Input data for training
The DANN takes haplotype matrices of "images" as input, where each row represents a pseudo-haplotype from a sample and 
each column the position of each variant site in the sample. These haplotype matrices have dimensions n x L, where we set n=150 
pseudo-haplotypes and L=201 SNPs in a given window. 
In these images, the color of each pixel represents the occurrence of the major or minor allele, where 
we transformed the alleles into binary values such that the major allele was coded as -1 and minor allele as 1. Missing data was coded as 0. 

### sorting haplotypes


# 2) Training DANN

Once the simulations have been generated and processed and the real aDNA has been processed we are ready to train the DANN.
To train the model we submit the job **qsub_TrainModel**, which is running the script **main_train.py**.

To run this script we have to define the following:
* **Model name.** This is the model name we use for the optput (e.g _model_name='GRL_multiclass'_).

  During training we will generate n+1 files where n= number of training epochs. The file _GRL_multiclass_model.json_ contains the model architechture and the remaining n files (_GRL_multiclass.1.weights.h5,GRL_multiclass.2.weights.h5,...GRL_multiclass.n.weights.h5_) contain the weights after each epoch of training.
* **path to training data.** We define four separate paths to our training data:
  - mmap_neutral = '../Data/sims/neutral_ConstantNeMD43_RowFreq_n150_w201_sims.npy' # neutral simulations
  - mmap_HS = '../Data/sims/HS_ConstantNeMD43_RowFreq_n150_w201_sims.npy' # Hard sweep processed simulations
  - mmap_SS = '../Data/sims/SS_ConstantNeMD43_RowFreq_n150_w201_sims.npy' # Soft sweeps processed simulations
  - mmap_target = '../Data/aDNA/target_N-H_ALL_RowFreq_n150_w201.npy' # real processed aDNA data

```bash
python main_train.py model_name mmap_neutral mmap_HS mmap_SS mmap_target
```

  within the _main_train.py_ script you can adjust the batch size (set as 64 by default)

