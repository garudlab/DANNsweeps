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
rho = np.random.uniform(3e-9,2e-8) #np.random.uniform(3e-9,2e-8) draw_truncated_exponential(1e-8, 3e-8)
mu = np.random.uniform(1e-8,1.5e-8)



command='slim ' +' -d R=' + str(rho)+ ' -d N='+ str(Ne)+ ' -d MU='+ str(mu)   
command += ' -d id='+str(id)+ ' -d run='+str(id)+str(i)
command += ' -d Q='+str(Q) +' Neutral_tree.slim'
print(command)

