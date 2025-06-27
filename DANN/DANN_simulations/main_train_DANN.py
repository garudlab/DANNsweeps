### --------- load modules -------------------#
import sys
import os

from GRL_multiclass_data_Simulations_A100 import *
import gc


# load data 
model_name=str(sys.argv[1])

#source
mmap_neutral_path = str(sys.argv[2])
mmap_HS_path = str(sys.argv[3])
mmap_SS_path = str(sys.argv[4])
#target 
model_name_tar=str(sys.argv[5])
mmap_neutral_path_tar = str(sys.argv[6])
mmap_HS_path_tar = str(sys.argv[7])
mmap_SS_path_tar = str(sys.argv[8])


### --------- Parameters -------------------#
initial_loss_weights=[1,0]
val_split=0.1
batch_size=64 #64, 32

### --------- Build and train model -------------------#

print("BUILDING MODEL")
model = create_model(mmap_neutral,loss_weights=initial_loss_weights)


#print(model.summary())

#train model
print("TRAINING MODEL")
model,score=train_model(model,mmap_neutral,mmap_HS,mmap_SS,mmap_neutral_tar,mmap_HS_tar,mmap_SS_tar,\
                        val_split=val_split,batch_size=batch_size,loss_weights=initial_loss_weights,path=model_name) 


