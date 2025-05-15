import msprime
import pyslim, tskit
import os.path
import numpy as np
import sys
import random


def mutate_and_recap(inFile, outFile,r,N,mu,Q):
      #n_window=201
      # Load the .trees file from slim output
      ts = tskit.load(inFile) #pyslim.load(inFile)
      ts = pyslim.update(ts)
      #recapitate
      rts= pyslim.recapitate(ts,recombination_rate=r*Q, ancestral_Ne=int(N/Q))
      
      #We check if recapitatoin worked by verifying if all trees have only one root
      orig_max_roots = max(t.num_roots for t in ts.trees())
      recap_max_roots = max(t.num_roots for t in rts.trees())      
      print(f"Maximum number of roots before recapitation: {orig_max_roots}\n"
      f"After recapitation: {recap_max_roots}")

      #simplify and sample
      individuals = pyslim.individuals_alive_at(rts, 0) #individuals alive at present
      sample = np.random.choice(individuals, size=177,replace=False)
      keep_nodes = []
      for i in sample:
           keep_nodes.extend(rts.individual(i).nodes)
      print("simplification")
      sts = rts.simplify(keep_nodes)

      print(f"Before, there were {rts.num_samples} sample nodes (and {rts.num_individuals} individuals)\n"
      f"in the tree sequence, and now there are {sts.num_samples} sample nodes\n"
      f"(and {sts.num_individuals} individuals).")

      #add mutations
      print("adding neutral mutations")
      mutated=msprime.sim_mutations(sts,rate=mu*Q,model=msprime.SLiMMutationModel(type=1),keep=True,)
      #mutated = pyslim.SlimTreeSequence(msprime.sim_mutations(sts,rate=mu*Q,model=msprime.SLiMMutationModel(type=1),keep=True,))


      print(f"The tree sequence now has {mutated.num_mutations} mutations, "
      f"and mean pairwise nucleotide diversity is {mutated.diversity()}, "
      f"and number of sites is {mutated.num_sites}.")


      n_window = mutated.num_sites
      print("window segregating sites: "+ str(n_window))

      ###output S and Pi ?
      #print(mutated.genotype_matrix().shape)
      print("output MS file")

      out_file = open(outFile,'w')

      out_file.write("//"+"\n")
      out_file.write("segsites: "+ str(n_window) + "\n")    

      #get positions of variant sites
      pos=[]
      for variant in mutated.variants():
            pos.append(int(variant.position))

      print("num positions: " + str(len(pos)))

      indx_subsample= sorted(random.sample(range(len(pos)), n_window))
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
                  variants=variants[:n_window]
            out_file.write( "\n"+ "".join(variants))
      


      out_file.close()


def main():

    inFile = sys.argv[1]
    outFile = sys.argv[2]
    r = float(sys.argv[3])
    N = float(sys.argv[4])
    mu = float(sys.argv[5])
    Q = float(sys.argv[6])

    print("mu= "+str(mu))   
    mutate_and_recap(inFile, outFile,r,N,mu, Q) 

#run main
if __name__ == '__main__':
    main()
