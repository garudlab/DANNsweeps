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
