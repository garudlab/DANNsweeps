# Training a CNN and a DANN using simulated labeled data for source and target domains.

We can train two models, a CNN and a DANN, using simulated data for benchmarking and testing the efficacy of the DANN in a simulated scenario. 

To train a **CNN** we submit the job
```bash
qsub qsub_TrainModel_CNN #set correct virtual environment
```
or run the scrip directly:

```bash
python main_train_CNN.py model_name mmap_neutral mmap_HS mmap_SS
```
We must define the path to the files where the simulated data that is processed and ready for training is saved (mmap_neutral mmap_HS mmap_SS). 

To train the **DANN** we can submit the job:
```bash
qsub qsub_TrainModel_DANN #set correct virtual environment
```

or run the python script directly:

```bash
python main_train_CNN.py model_name mmap_neutral mmap_HS mmap_SS mmap_neutral_tar mmap_HS_tar mmap_SS_tar
```

Note that the scripts _CNN_multiclass_data_mergeSims_A100.py_ and _GRL_multiclass_data_Simulations_A100.py_ have to be in the same directory 
or we have to set the path to the directory where these are saved. These scripts contain all functions for training the models.


## Compare model performance

After training, select the model weights from the epoch with lowest validation loss. This can be found in the output file _training_multiclass_results.txt_ where the accuracy and loss for each epoch of training is found.

Once the correct model has been selected make sure to rename the weights file for the final model with the correct weights. For example if after training the DANN we 
find that the epoch with lowest validation is epoch 14 we can do
```bash
cp GRL_multiclass.14.weights.h5 cp GRL_multiclass.weights.h5
```
such that you end up with two files _GRL_multiclass.weights.h5_ and _GRL_model.json_

For the CNN model you should do the same to obtain two files with all the information from the model: _CNN_multiclass.weights.h5_ and _CNN_model.json_

Once you have selected the weights from each of your models and have defined the files accordingly we can compare the performance of the models running the following script:

```bash
python PR_curves_multiclass.py model_name_CNN model_name_DANN path_src_neu path_src_hs path_src_ss path_tar_neu path_tar_hs path_tar_ss
```

where 

```bash
model_name_CNN='CNN_multiclass'
model_name_DANN='DANN_multiclass'


path_src="/u/project/ngarud/Garud_lab/DANN/aDNA/ProcessingData/ProcessedDataALL/SortByRowFreq/"
path_tar="/u/project/ngarud/Garud_lab/DANN/aDNA/ProcessingData/ProcessedDataALL/SortByRowFreq/"

#path to test data
path_src_neu= path_src+"Neutral/ConstantNeMD43_Neutral_THETA1to5_n150_w201_RowFreq_trainsrc_50.dat"
path_src_hs= path_src+"HardSweeps/ConstantNeMD43_HS_THETA1to5_n150_w201_RowFreq_trainsrc_50.dat"
path_src_ss= path_src+"SoftSweeps/ConstantNeMD43_SS_THETA1to5_n150_w201_RowFreq_trainsrc_50.dat"

path_tar_neu= path_tar+"Neutral/SouilmiMD43_Neutral_THETA1to5_n150_w201_RowFreq_trainsrc_50.dat" #SouilmiMD43
path_tar_hs= path_src+"HardSweeps/SouilmiMD43_HS_THETA1to5_n150_w201_RowFreq_trainsrc_50.dat"
path_tar_ss= path_src+"SoftSweeps/SouilmiMD43_HS_THETA1to5_n150_w201_RowFreq_trainsrc_50.dat"
```



