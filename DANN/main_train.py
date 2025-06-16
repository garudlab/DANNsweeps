### --------- load modules -------------------#
import sys
import os

from GRL_multiclass_data_mergeSims_A100 import * # import model and functions for training
import gc

# load data 
model_name=str(sys.argv[1])
mmap_neutral_path = str(sys.argv[2])
mmap_HS_path = str(sys.argv[3])
mmap_SS_path = str(sys.argv[4])
mmap_target_path = str(sys.argv[5])

mmap_neutral = np.load(mmap_neutral_path, mmap_mode='r') 
mmap_HS = np.load(mmap_HS_path, mmap_mode='r') 
mmap_SS = np.load(mmap_SS_path, mmap_mode='r')
mmap_target = np.load(mmap_target_path, mmap_mode='r') 


### --------- Parameters -------------------#
initial_loss_weights=[1,0] #[1,0]
val_split=0.1
batch_size=64 #64,32


### --------- Build and train model -------------------#

print("BUILDING MODEL")
model = create_model(mmap_neutral,loss_weights=initial_loss_weights)

#train model
print("TRAINING MODEL")
model,score=train_model(model,mmap_neutral,mmap_HS,mmap_SS,mmap_target,val_split=val_split,batch_size=batch_size,loss_weights=initial_loss_weights,path=model_name) #GRL_HS_DuringBottleneck_trained


