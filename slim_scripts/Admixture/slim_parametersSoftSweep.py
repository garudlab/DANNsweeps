import numpy as np
import sys
#from scipy.stats import rv_discrete
import os.path

rho_in=float(sys.argv[1])
Ne=float(sys.argv[2])
Mu=float(sys.argv[3])
id=int(sys.argv[4])
i=int(sys.argv[5])
Q=float(sys.argv[6]) 
program=sys.argv[7]
burnIN=10.0
ChrType='A'
sexRatio=0.5
H=0.5

#hyperparameters
s=np.random.uniform(0.01,0.1)
PF=np.random.uniform(0.5,0.95)
SweepStart=np.random.uniform(0.08,0.25)
adaptive_theta=np.random.uniform(1,3.0)

command='slim ' +' -d R=' + str(rho_in)+' -d burn_in='+ str(burnIN) + ' -d N='+ str(Ne)+ ' -d MU='+ str(Mu)   
command += ' -d sb='+str(s)+ ' -d PF='+str(PF)+ ' -d id='+str(id)+ ' -d run='+str(id)+str(i)
command += ' -d sexRatio='+str(sexRatio)+' -d H='+str(H)+' -d Q='+str(Q)+' -d SweepStart='+str(SweepStart)
command += ' -d THETA_A='+str(adaptive_theta)+' SoftSweep_tree.slim'
print(command)

