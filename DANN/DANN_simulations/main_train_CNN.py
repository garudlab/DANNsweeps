### --------- load modules -------------------#
import sys
import os

from CNN_multiclass_data_mergeSims_A100 import *
import gc


# load data 
model_name=str(sys.argv[1])
mmap_neutral_path = str(sys.argv[2])
mmap_HS_path = str(sys.argv[3])
mmap_SS_path = str(sys.argv[4])


### LOAD DATA
mmap_neutral = np.load(mmap_neutral_path, mmap_mode='r') 
mmap_HS = np.load(mmap_HS_path, mmap_mode='r') 
mmap_SS = np.load(mmap_SS_path, mmap_mode='r')



### --------- Parameters -------------------#
initial_loss_weights=[1,0]
val_split=0.1
batch_size=64 #64, 32

### --------- Build and train model -------------------#

print("BUILDING MODEL")
model = create_model(mmap_neutral)


print(model.summary())

#train model
print("TRAINING MODEL")
model,score=train_model(model,mmap_neutral,mmap_HS,mmap_SS,val_split=val_split,batch_size=batch_size,path=model_name) 


