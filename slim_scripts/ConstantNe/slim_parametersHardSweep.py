import numpy as np
import sys
#from scipy.stats import rv_discrete
import os.path

Ne=float(sys.argv[1])
id=int(sys.argv[2])
i=int(sys.argv[3])
Q=float(sys.argv[4]) 
program=sys.argv[5]

#hyperparameters
rho = np.random.uniform(3e-9,2e-8)
mu = np.random.uniform(1e-8,1.5e-8)
s=np.random.uniform(0.01,0.1)
PF=np.random.uniform(0.5,0.95)
SweepStart=np.random.uniform(0.08,0.25)
adaptive_theta=0.01


command='slim ' +' -d R=' + str(rho) + ' -d N='+ str(Ne)+ ' -d MU='+ str(mu)   
command += ' -d sb='+str(s)+ ' -d PF='+str(PF)+ ' -d id='+str(id)+ ' -d run='+str(id)+str(i)
command += ' -d Q='+str(Q)+' -d SweepStart='+str(SweepStart)
command += ' -d THETA_A='+str(adaptive_theta)+' HardSweep_tree.slim'
print(command)

