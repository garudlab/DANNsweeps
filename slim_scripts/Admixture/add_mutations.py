import msprime
import pyslim, tskit
import os.path
import numpy as np
import sys
import random


def overlay_neutral_muts(inFile, outFile,mu,Q):
      n_window = 201
      # Load the .trees file from slim output
      ts = tskit.load(inFile) #pyslim.load(inFile)
      ts = pyslim.update(ts)
            
      #We check if recapitatoin worked by verifying if all trees have only one root
      max_roots = max(t.num_roots for t in ts.trees())   
      print(f"Maximum number of roots: {max_roots}")

      #simplify and sample
      individuals = pyslim.individuals_alive_at(ts, 0, population=0) #individuals alive at present
      sample = np.random.choice(individuals, size=177,replace=False)
      keep_nodes = []
      for i in sample:
           keep_nodes.extend(ts.individual(i).nodes)
      print("simplification")
      sts = ts.simplify(keep_nodes)

      print(f"Before, there were {ts.num_samples} sample nodes (and {ts.num_individuals} individuals)\n"
      f"in the tree sequence, and now there are {sts.num_samples} sample nodes\n"
      f"(and {sts.num_individuals} individuals).")

      #add mutations
      print("adding neutral mutations")
      mutated=msprime.sim_mutations(sts,rate=mu*Q,model=msprime.SLiMMutationModel(type=1),keep=True,)
      
      print(f"The tree sequence now has {mutated.num_mutations} mutations, "
      f"and mean pairwise nucleotide diversity is {mutated.diversity()}, "
      f"and number of sites is {mutated.num_sites}.")

      ###output S and Pi ?
      #print(mutated.genotype_matrix().shape)
      print("output MS file")

      out_file = open(outFile,'w')

      out_file.write("//"+"\n")
      #out_file.write("segsites: "+ str(mutated.num_sites) + "\n")
      out_file.write("segsites: "+ str(n_window) + "\n")

      #get positions of variant sites
      pos=[]
      for variant in mutated.variants():
            pos.append(int(variant.position))

      indx_subsample= sorted(random.sample(range(len(pos)), 201))
      pos_sub = [pos[i] for i in indx_subsample]
      #add positions to MS output
      pos_lst=[str(element / 4.5e5) for element in pos_sub]
      pos_str= [str(element) for element in [pos_sub]]
      out_file.write("positions: " + " ".join(pos_lst))

      Geno_matrix=mutated.genotype_matrix()

      Geno_matrix_sub = Geno_matrix[indx_subsample,::2]
      print(Geno_matrix_sub.shape)
      
      num_cols=Geno_matrix_sub.shape[1]
      print("Matrix shape:", Geno_matrix_sub.shape)

      for col in range(num_cols):
            variants = [1 if x > 1 else x for x in Geno_matrix_sub[:,col]]
            variants= [str(element) for element in variants]
            if len(variants) != n_window:
                  print("ERROR -- wrong number of variants")
                  variants.pop(n_window//2) #remove element from center (comes from >9 num origins so repeated site)
                  
            out_file.write( "\n"+ "".join(variants))
      


      out_file.close()


def main():

    inFile = sys.argv[1]
    outFile = sys.argv[2]
    mu = float(sys.argv[3])
    Q = float(sys.argv[4])

    print("mu= "+str(mu))   
    overlay_neutral_muts(inFile, outFile,mu, Q) 

#run main
if __name__ == '__main__':
    main()
