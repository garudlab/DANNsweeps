# DANNsweeps

Code for manuscript: Harris M, Mo Z, Siepel A and Garud N. _The persistence and loss of hard selective sweeps amid ancient human admixture_ (https://www.biorxiv.org/content/10.1101/2025.10.14.682443v1).

## 1) Set up
To set up the Hoffman2 UCLA cluster for training the model on a A100 GPU node run steps in **install_tensorflowGPU_documentation**

To install all requirements activate the conda environment and install requirements found in _requirements.txt_

```bash
 conda activate tf_gpu_A100
 pip install -r requirements.txt
```

## 2) Data
All scripts to run simulations used for source domain and for simulation study are found in **slim scripts/** directory.

## 3) Training
Code to train the DANN is found in the **DANN/** directory.

Code to train the DANN and a CNN using simulated data only can be found in **DANN/DANN_simulations/** 

