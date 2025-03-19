import msprime
import pyslim, tskit
import os.path
import numpy as np
import sys

# ---------- variables ------------------------
id=int(sys.argv[1])
i=int(sys.argv[2])
Q=float(sys.argv[3])
program=sys.argv[4]
generation_time=29
# ---------------------------------------------
def draw_truncated_exponential(mean, truncation):
    while True:
        sample = np.random.exponential(scale=mean)
        if sample <= truncation:
            return sample
# ---------- hyperparameters ------------------
H=0.5
chr_len=450000
rho = np.random.uniform(3e-9,2e-8) #np.random.uniform(3e-9,2e-8) draw_truncated_exponential(1e-8, 3e-8)
mu = np.random.uniform(1e-8,1.5e-8)
s=np.random.uniform(0.01,0.1)
sweep_gbp = round(np.random.uniform(840,1500)) #sweep start time # generations ago
PF=np.random.uniform(0.5,0.95)
SweepStart=np.random.uniform(0.08,0.25)
#adaptive_theta=round(np.random.uniform(1.0,5.0))

file_burn_in='tmp_intermediate_files/Souilmi_burn_in_'+str(id)+str(i)+'.trees'
# ---------------------------------------------

# ---------- burn in ------------------

t_=600200//generation_time


demog_model = msprime.Demography()

demog_model.add_population(initial_size=18200)
demog_model.add_population_parameters_change(time=t_,  initial_size=29100)

ots = msprime.sim_ancestry(
        samples=29100,
        demography=demog_model,
        sequence_length=chr_len,
        recombination_rate=rho)

ots = pyslim.annotate(ots, model_type="WF", tick=1, stage="late")

tables = ots.tables

ots.dump(file_burn_in)

# ---------------------------------------------

# ---------- slim command ------------------

command='slim ' +' -d recomb_rate=' + str(rho)+ ' -d MU='+ str(mu) + ' -d "trees_BurnIN_file='+"'"+file_burn_in+"'"+'"'
command += ' -d sb='+str(s)+ ' -d PF='+str(PF)+ ' -d id='+str(id)+ ' -d run='+str(id)+str(i)
command += ' -d H='+str(H)+' -d SweepStart='+str(SweepStart)+' -d sweep_gbp='+str(sweep_gbp)
command += ' Souilmi_Eurasia_postBurnIn_HS.slim'
print(command)

#' -d "file_out='+"'"+'SLiMout/VCF_K'+str(K)+'_introMut'+str(introMut)+'.csv'+"'"+'"'
