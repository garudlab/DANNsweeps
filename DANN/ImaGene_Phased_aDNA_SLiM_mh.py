### --------- load modules -------------------#
import os
import gzip
import _pickle as pickle

import numpy as np
import scipy.stats
import math
import random


import itertools
import skimage.transform
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.metrics import confusion_matrix
#import pydot # optional, but required by keras to plot the model

### ------------- utilities --------------------
def clusterHaps(numSamples,samples_dict):
    # count haps  create dictionary where key = hap and value = sample id
    haps={}
    for j in range(numSamples):
        hap_array = ','.join(map(str,samples_dict[j][:,0]))
        haps.setdefault(hap_array,[])
        haps[hap_array].append(j)
    
    #now clump haplotypes
    #If a haplotype matches another haplotype at all positions except for sites where there are Ns (missing data), then the haplotypes will be combined and the 'distance' between the two haplotypes will be considered 0.
    haps_clumped ={} #all clumped haplotypes
    haps_clumped_count={} # count number of haplotypes that are clumped

    compared ={} # keep track of the haps I have compared
    for key1 in haps.keys():
        if (key1 in compared) == False: # check if I've already compared this hap
            compared[key1]=1
            haps_clumped[key1] = haps[key1]
            haps_clumped_count[key1] = 1
            
            for key2 in haps.keys(): #iterate across other haplotype to compare to
                if ((haps[key2][0] in haps_clumped[key1])== False) and ((key2 in compared) == False): #check if sample is not already included in ha[s_clumped and that we have not iterated over key2
                    [distance,s1] = hamming_distance_clump(key1,key2,0.75)
                    if distance == 0 and key1 != s1: # if I replaced 'nan' in key1, I will replace the returned key1 in haps clumped
                        haps_clumped_count[s1] = haps_clumped_count[key1]
                        haps_clumped[s1] =haps_clumped[key1]
                        del haps_clumped_count[key1]
                        del haps_clumped[key1]
                        key1=s1
                    
                    if distance <= 0:#less  or equal to distance threshold. Distance is 0 and key1 could be == s1 (i.e missing data in s2 is merged with s1)
                        haps_clumped[key1] += haps[key2] # add the array for key2 to key1 array
                        haps_clumped_count[key1] += 1
                        compared[key2] = 1 # I won't check this distance again since it has been clumped
    
    # Create a new dictionary with lengths of values
    haps_clump_adjusted = {key: len(value) for key, value in haps_clumped.items()}

    return [haps_clumped,haps_clump_adjusted]
                    
def clusterHaps_byDistance(haps_clumped_count_sort,missing_thresh):
    haps_clumped_distance= {}
    compared = {}
    hap1 = max(haps_clumped_count_sort, key=haps_clumped_count_sort.get)
    compared[hap1]=1 # most frequent haplotype
    haps_clumped_distance[hap1]=0 # cero distance to most common hap
    for key in haps_clumped_count_sort.keys():
        if (key in compared) == False:
            # If I havent compared calc distance
            list_s1 = hap1.split(',')
            list_s2 = key.split(',')
            #count nan's
            numNaN_s1 = list_s1.count('nan')
            numNaN_s2 = list_s2.count('nan')
            if numNaN_s1 >= int(len(list_s1)*missing_thresh) or numNaN_s2 >= int(len(list_s2)*missing_thresh): #if hap has more than missing_thresh missing dat
                distance=sum(x != y and (x != 'nan' or y != 'nan') for x, y in zip(list_s1, list_s2)) # if > missing thresh nan's count as differences 
                #distance =len(list_s1)
            else:
                distance = 0
                for i in range(len(list_s1)):
                    if list_s1[i] != list_s2[i]:
                        if list_s2[i] != 'nan':
                            if list_s1[i] != 'nan':
                                distance +=1
            haps_clumped_distance[key]= distance
    return haps_clumped_distance

def hamming_distance_clump(s1,s2,missing_thresh):
        list_s1 = s1.split(',')
        list_s2 = s2.split(',')
        #count nan's
        numNaN_s1 = list_s1.count('nan')
        numNaN_s2 = list_s2.count('nan')
        if numNaN_s1 >= int(len(list_s1)*missing_thresh) or numNaN_s2 >= int(len(list_s2)*missing_thresh): #if hap has more than missing_thresh missing dat
            distance =len(list_s1)
        else:
            distance = 0
            for i in range(len(list_s1)):
                if list_s1[i] != list_s2[i]:
                    if list_s2[i] != 'nan':
                        if list_s1[i] != 'nan':
                            distance +=1
                            if distance > 0: #distance threshold
                               return [distance, s1]
                        else:
                            s1 = ','.join(list_s1[:i] + [list_s2[i]] + list_s1[i+1:])

        return [distance,s1]

def to_binary(targets):
    return np.asarray(np.where(targets == targets.min(), 0, 1).astype('float32'))


def to_categorical(targets, wiggle=0, sd=0):
    classes = np.unique(targets)
    nr_classes = len(classes)
    print(nr_classes)
    results = np.zeros((len(targets), len(classes)), dtype='float32')
    for counter, value in enumerate(targets):
        index = np.where(classes == value)[0]
        # add wiggle (if any)
        if wiggle > 0:
            index += np.random.randint(low=-wiggle, high=wiggle+1)
            if index < 0:
                index = 0
            elif index >= results.shape[1]:
                index = results.shape[1] - 1
        results[counter, index] = 1.
        # add sd (if any)
        if sd > 0:
            probs = scipy.stats.norm.pdf(range(nr_classes), loc=index, scale=sd)
            results[counter, ] = probs / probs.sum()
            del probs
    return results

def load_imagene(file):
    """
    Load ImaGene object
    """
    with open(file, 'rb') as fp:
        gene = pickle.load(fp)
    return gene

def load_imanet(file):
    """
    Load ImaNet object
    """
    with open(file, 'rb') as fp:
        net = pickle.load(fp)
    return net


### -------- objects ------------------



class ImaFile:
    """
    Parser for real data and simulations
    """
    def __init__(self, nr_samples, simulations_folder=None, VCF_file_name=None,G12_file_name=None, model_name='N/A'):
        self.simulations_folder = simulations_folder
        self.nr_samples = nr_samples
        self.VCF_file_name = VCF_file_name
        self.G12_file_name = G12_file_name
        self.model_name = model_name
        return None

    def extract_description(self, file_name,numSims):


        """
        ### Note: this works for msms output where first line has the parameters from simulation but not for SLiM output
        I changed this code to parse SLiM output
        it is based on the file name which should be of the form
        Text_THETA_target_sim.txt
        with Ne= effective population size,
     
        ### WHat will I do with different models i.e. bottlenecks etc?????   
        
        Keyword Arguments:
            file_name (string) -- name of simulation file
            model_name (string) -- name of demographic model

        Return:
            description (string)
        """
        file_name_lst=file_name.split('_')
        THETA=float(file_name_lst[1].replace("THETA", ""))
        target=float(file_name_lst[2].replace("target", "")) #1 if simulations/data comes from target domain, 0 if it comes from source domain
        #print(file_name)
        desc = {'name':file_name_lst[0].split('/')[-1]}
        
        # Extracting parameters 
        #These are the ones that can be inferred 
        desc.update({'nr_chroms':int(177)}) #number of samples
        desc.update({'nr_replicates':int(numSims)}) #num sites

        desc.update({'mutation_rate':float(1e-9)})
        desc.update({'recombination_rate':float(5e-9)})
        desc.update({'selection_position':float(0.5)}) 

        desc.update({'THETA':float(THETA)})
        desc.update({'target':float(target)})

        desc.update({'model':str(self.model_name)})

        # Get the UNIX Time Stamp of when the file was modification
        desc.update({'modification_stamp':os.stat(file_name).st_mtime})

        # Allow deleted files to be tracked in json folder
        desc.update({'active':'active'})


        return desc

    def read_simulations(self, classifier_name='THETA',discriminator_name='target', max_nrepl=None, verbose=0):
        """
        Read simulations and store into compressed numpy arrays

        Keyword Arguments:
            classifier_name: name of parameter(s) to estimate 
            discriminator_name: name of parameter to discriminate between source versus target domains
            max_nrepl: max nr of replicates per simulated msms file
            verbose: 


        Returns:
            an object of class Genes
        """

        '''
        Edited this code to process SLiM simulations
        '''

        data = []
        positions = []
        description = []

        # Open the directory in which simulation files are stored
        #for file_name in os.listdir(self.simulations_folder):

            #full_name = self.simulations_folder + '/%s' %(file_name)
        full_name = self.simulations_folder
        if verbose > 0:
            print(full_name, ': ', end='')

        # Read lines including the metadata
        if full_name.endswith(".gz"):
            f = gzip.open(full_name, 'rb')
        else:
            f = open(full_name, 'rb')

        file_content = f.read().decode('utf8').split('\n')


        # Search the // char inside the file
        starts = ([i for i, e in enumerate(file_content) if e == '//'])

        numSims=len(starts) #number of simulations
        
        # limit the scan to the first max_nrepl items (if set)
        if max_nrepl!=None:
            starts = starts[:max_nrepl]

        if verbose > 0:
            print(len(starts))

        # Populate object with data for each simulated gene
        #description.append(full_name)
        #print(full_name.split('_'))
        #print(starts)
        for idx, pointer in enumerate(starts):

            # Description for each simulation
            description.append(self.extract_description(full_name,numSims))

            nr_columns = int(file_content[pointer+1].split('segsites: ')[1])
            haplotypes = np.zeros((self.nr_samples, nr_columns, 1), dtype='object')
            pos = file_content[pointer+2].split(' ')
            #pos.pop()  #what was this for????
            pos.pop(0)
            
            positions.append(np.asarray(pos, dtype='float32'))
            del pos

            for j in range(self.nr_samples):

                hap = list(file_content[pointer + 3 + j])
                
                # string processing: if not 0/1 --> convert to 1
                #hap = [1.0 if element!='0' and element!='1' and element!='N' else element for element in hap]
                hap = ['1' if element!='0' and element!='1' and element!='N' else element for element in hap]
                # switch colours, 1s are black and 0s are white
                #hap = [255.0 if element==1.0  or element =='1' else element for element in hap]
                hap = [255.0 if element =='1' else element for element in hap]

                hap = [0.0 if element=='0' else element for element in hap]
                # gray for missing data
                hap = [np.nan if element=='N' else element for element in hap] #Value for missing data 200?
          
                try:
                    haplotypes[j,:,0] = hap
                except ValueError as e:
                    sites_len=haplotypes.shape[1]
                    if len(hap)> sites_len:
                        hap=hap[:sites_len]
                        haplotypes[j,:,0] = hap
                    else:
                        print(e)
                    
            data.append(haplotypes) 
            
            f.close()

        gene = ImaGene(data=data, positions=positions, description=description, classifier_name=classifier_name,discriminator_name=discriminator_name)

        return gene

    def read_VCF(self, verbose=0):
        """
        Read VCF file and store into compressed numpy arrays

        Keyword Arguments:
            verbose: 

        Returns:
            an object of class Genes
        """

        with open(self.VCF_file_name, 'r') as f:
            lines = [l for l in f if not l.startswith('##')]

        header = lines.pop(0)
        ind_pos = header.split('\t').index('POS')
        ind_format = header.split('\t').index('FORMAT')

        nr_individuals = len(header.split('\t')) - ind_format - 1
        nr_sites = len(lines)

        if verbose == 1 | self.nr_samples!=(nr_individuals*2):
            print('Found' + str(nr_individuals) + 'individuals and' + str(nr_sites) + 'sites.')

        #haplotypes = np.zeros(((nr_individuals * 2), nr_sites, 1), dtype='uint8')
        haplotypes = np.zeros(((nr_individuals), nr_sites, 1), dtype='object')
        data = []
        positions = []
        pos = np.zeros((nr_sites), dtype='int32')
        for j in range(nr_sites):
            # populate genomic position
            pos[j] = int(lines[j].split('\t')[ind_pos])
            # extract genotypes
            genotypes = lines[j].split('\t')[(ind_format+1):]
            genotypes[len(genotypes) - 1] = genotypes[len(genotypes) - 1].split('\n')[0]
            for i in range(len(genotypes)):
                if i == 0:
                    i1 = 0
                    i2 = 1
                else:
                    i2 =  i #i*2
                    i1 = i2 - 1
                pseudo_hap = random.choice(genotypes[i].split('/'))
                if pseudo_hap == '1':
                    haplotypes[i1,j] = 255.0 #white?
                if pseudo_hap == '0':
                    haplotypes[i1,j] = 0.0 #white?
                if pseudo_hap == '.':
                    haplotypes[i1,j] = np.nan

        positions.append(pos)
        data.append(haplotypes)
        del pos
        del haplotypes
        gene = ImaGene(data=data, positions=positions)
        return gene
    
    def read_G12format_file(self, verbose=0):
        with open(self.G12_file_name, 'r') as f:
            lines = [l for l in f]
        
        nr_individuals = len(lines[0].split(','))  - 1
        nr_sites = len(lines)

        if verbose == 1:
            print('Found' + str(nr_individuals) + 'samples and' + str(nr_sites) + 'sites.')
        
        haplotypes = np.zeros(((nr_individuals), nr_sites, 1), dtype='object')
        data = []
        positions = []
        pos = np.zeros((nr_sites), dtype='int32')
        for j in range(nr_sites):
            # populate genomic position
            pos[j] = int(lines[j].split(',')[0])
            genotypes = lines[j].split(',')[1:]
            genotypes[len(genotypes) - 1] = genotypes[len(genotypes) - 1].split('\n')[0]
            alleles = [x for x in set(genotypes) if x != 'N']
            for i in range(len(genotypes)):
                if genotypes[i] == 'N':
                    haplotypes[i,j] = np.nan
                if len(alleles) > 0 and genotypes[i] == alleles[0]:
                    haplotypes[i,j] = 0.0
                if len(alleles) > 1 and genotypes[i] == alleles[1]:
                    haplotypes[i,j] = 255.0
        positions.append(pos)
        data.append(haplotypes)
        del pos
        del haplotypes
        gene = ImaGene(data=data, positions=positions)
        return gene

class ImaGene:
    """
    A batch of genomic images
    """
    def __init__(self, data, positions, description=[], targets_classifier=[],targets_discriminator=[], classifier_name=None,discriminator_name=None, classes_classifier=[], classes_discriminator=[]):
        self.data = data
        self.positions = positions
        self.description = description
        self.dimensions = (np.zeros(len(self.data)), np.zeros(len(self.data)))
        # initialise dimensions to the first image (in case we have only one)
        self.dimensions[0][0] = self.data[0].shape[0]
        self.dimensions[1][0] = self.data[0].shape[1]
        if len(self.data)> 1: # if we have nire thab one image
            for i in range(len(self.data)):
                # assign dimensions
                self.dimensions[0][i] = self.data[i].shape[0]
                self.dimensions[1][i] = self.data[i].shape[1]    
        # if reads from real data, then stop here otherwise fill in all info on simulations
        if classifier_name != None:
            self.classifier_name = classifier_name # this is passed by ImaFile.read_simulations()
            self.targets_classifier = np.zeros(len(self.data), dtype='float32') #int32
            for i in range(len(self.data)):
                # set targets from file description
                self.targets_classifier[i] = self.description[i][self.classifier_name]
                # assign dimensions
                self.dimensions[0][i] = self.data[i].shape[0]
                self.dimensions[1][i] = self.data[i].shape[1]    
            self.classes_classifier = np.unique(self.targets_classifier)
        #now do the same for discriminator classes
        if discriminator_name != None:
            self.discriminator_name = discriminator_name # this is passed by ImaFile.read_simulations()
            self.targets_discriminator = np.zeros(len(self.data), dtype='float32') #int32
            print(len(self.data))
            for i in range(len(self.data)):
                # set targets from file description
                self.targets_discriminator[i] = self.description[i][self.discriminator_name]
                # assign dimensions
                self.dimensions[0][i] = self.data[i].shape[0]
                self.dimensions[1][i] = self.data[i].shape[1]    
            self.classes_discriminator = np.unique(self.targets_discriminator)
            
        return None

    def summary(self):
        """
        Prints general info on the object.

        Keyword Arguments:


        Returns:
            0
        """
        nrows = self.dimensions[0]
        ncols = self.dimensions[1]
        print('An object of %d image(s)' % len(self.data))
        print('Rows: min %d, max %d, mean %f, std %f' % (nrows.min(), nrows.max(), nrows.mean(), nrows.std()))
        print('Columns: min %d, max %d, mean %f, std %f' % (ncols.min(), ncols.max(), ncols.mean(), ncols.std()))
        return 0

    def plot(self, index=0):
        """
        Plot one image in gray scale.

        Keyword arguments:
            index: index of image to plot

        Returns:
            0
        """
        replace_nan = [[2 if math.isnan(value) else value for value in row] for row in self.data[index][:,:,0]] # 0.5 will be gray, >1 MD will be white
        replace_nan =np.array(replace_nan)
        
        custom_cmap = ListedColormap(['black', 'red','white'])
        image = plt.imshow(replace_nan, cmap=custom_cmap) #cmap='gray'
        plt.show(image)
        return 0

    def majorminor(self):
        """
        Convert to major/minor polarisation.

        Keyword Arguments:

        Returns:
            0
        """
        for i in range(len(self.data)):
            data=self.data[i][:,:,0].astype(float)
            idx = np.where(np.nanmean(data/255., axis=0) > 0.5)[0]
            #idx = np.where(np.nanmean(self.data[i][:,:,0]/255., axis=0) > 0.5)[0]
            self.data[i][:,idx,0] = 255 - self.data[i][:,idx,0]
        return 0

    def sampleHaps(self,n,verbose=0):
        """
        Taples n haps with least missing data per sample

        Keyword Arguments:

        Returns:
            0
        """
        for i in range(len(self.data)):
            data=self.data[i][:,:,0].astype(float)
            # Calculate the number of missing values in each row
            missing_counts = np.isnan(data).sum(axis=1)
            # Get the indices of the rows with the least missing data
            sorted_idx = np.argsort(missing_counts)
            # Select the top 100 rows with the least missing data
            selected_idx = sorted_idx[:n]
            self.data[i] = self.data[i][selected_idx]
            self.dimensions[0][i] = self.data[i].shape[0]
            self.dimensions[1][i] = self.data[i].shape[1]



            subsampled_data = data[selected_idx]
            #print(self.data[i].shape)
            #self.data[i][:,:,0]= subsampled_data
            #print(subsampled_data.shape)


            # update based on index
            #self.data = self.data[index]
            #self.positions = [self.positions[i] for i in index]
            #self.description = [self.description[i] for i in index]
            #for i in range(len(self.data)):
            #    self.dimensions[0][i] = self.data[i].shape[0]
            #    self.dimensions[1][i] = self.data[i].shape[1]


        return 0

    def filter_freq(self, minimal_maf, verbose=0):
        """
        Remove sites whose minor allele frequency is below the set threshold.

        Keyword Arguments:
            minimal_maf: minimal minor allele frequency to retain the site

        Returns:
            0
        """
        for i in range(len(self.data)):
            data=self.data[i][:,:,0].astype(float)
            idx = np.where(np.nanmean(data/255., axis=0) >= minimal_maf)[0]
            #idx = np.where(np.nanmean(self.data[i][:,:,0]/255., axis=0) >= minimal_maf)[0]
            self.positions[i] = self.positions[i][idx]
            self.data[i] = self.data[i][:,idx,:]
            # update nr of columns in dimensions
            self.dimensions[1][i] = self.data[i].shape[1]
        return 0

    def sort_centerWindow(self, sub_length, ordering):
        """
        Sort center subwindow using row freq sorting and then sort each haplotype group from full window using 
        row)dist sorting

        Keyword Arguments:
            sub_length - subwindow length i.e 51 SNPS
            ordering -- ordering approach to use, sort central window by haplotype frequencies or by distance to most common hap

        Returns:
            0
        """
        for i in range(len(self.data)): 
            numSamples=self.data[i].shape[0]
            samples_dict = {} #empty dict
            for j, row in enumerate(self.data[i]):
                samples_dict[j] = row

            #get center window of length sub_length
            y=self.data[i].shape[1]
            starty = y // 2 - sub_length // 2
            data_sub=self.data[i][:, starty:starty + sub_length, :]

            [haps_clumped, haps_clumped_count] =clusterHaps(numSamples,data_sub)
              
            if ordering == 'rows_freq': # sort central window by frequency of haplotype
                haps_clumped_count_sort =dict(sorted(haps_clumped_count.items(), key=lambda item: item[1], reverse=True))
                gene_data_sorted_lst=[]
                for hap in haps_clumped_count_sort.keys():
                    # next I want to get all the samples in the full window with subhaplotype hap and order acording to distance to most common hap
                    ids_hap=haps_clumped[hap]
                    data_hap=self.data[i][ids_hap,:,:] #subset of samples of type hap in full window
                    #print(ids_hap)
                    [haps_clumped_window, haps_clumped_count_window] =clusterHaps(data_hap.shape[0],data_hap) # this is giving me line numbers based on subhap
                    #sort by row dist now
                    haps_clumped_count_window_sort =dict(sorted(haps_clumped_count_window.items(), key=lambda item: item[1], reverse=True))
                    #now order by distance to hap of highest freq
                    haps_clumped_window_distance = clusterHaps_byDistance(haps_clumped_count_window_sort,0.75)
                    #sort from least to greatest distance
                    haps_clumped_window_distance_sort = dict(sorted(haps_clumped_window_distance.items(), key=lambda item: item[1], reverse=False))
                    #now sort data
                    for hap_window in haps_clumped_window_distance_sort.keys():
                        hap_sub_sorted_ids=[ids_hap[i] for i in haps_clumped_window[hap_window]] 
                        for id in hap_sub_sorted_ids: # iterate across all samples that have haplotype hap
                            gene_data_sorted_lst.append(samples_dict[id])                           
                gene_data_sorted = np.array(gene_data_sorted_lst)
                self.data[i][:,:,:] = gene_data_sorted
        
            elif ordering == "rows_dist":
                haps_clumped_count_sort =dict(sorted(haps_clumped_count.items(), key=lambda item: item[1], reverse=True))
                #now order by distance to hap of highest freq
                haps_clumped_distance = clusterHaps_byDistance(haps_clumped_count_sort,0.75)
                #sort from least to greatest distance
                haps_clumped_distance_sort = dict(sorted(haps_clumped_distance.items(), key=lambda item: item[1], reverse=False))
                gene_data_sorted_lst=[]
                for hap in haps_clumped_distance_sort.keys():
                    # next I want to get all the samples in the full window with subhaplotype hap and order acording to frequency
                    ids_hap=haps_clumped[hap]
                    data_hap=self.data[i][ids_hap,:,:] #subset of samples of type hap in full window
                    
                    [haps_clumped_window, haps_clumped_count_window] =clusterHaps(data_hap.shape[0],data_hap) # this is giving me line numbers based on subhap
                    haps_clumped_count_window_sort =dict(sorted(haps_clumped_count_window.items(), key=lambda item: item[1], reverse=True))
                    #now sort data
                    for hap_window in haps_clumped_count_window_sort.keys():
                        hap_sub_sorted_ids=[ids_hap[i] for i in haps_clumped_window[hap_window]] 
                        for id in hap_sub_sorted_ids: # iterate across all samples that have haplotype hap
                            gene_data_sorted_lst.append(samples_dict[id])
                gene_data_sorted = np.array(gene_data_sorted_lst)
                self.data[i][:,:,:] = gene_data_sorted

            else:
                print('Select a valid ordering.')
                return 1

        return 0


    def sort(self, ordering):
        """
        Sort rows and/or columns given an ordering.

        Keyword Arguments:
            ordering: either 'rows_freq', 'rows_dist' -- No sorting by columns at the moment

        Returns:
            0
        """

        for i in range(len(self.data)): #range(len(self.data)):
            numSamples=self.data[i].shape[0]
            samples_dict = {} #empty dict
            for j, row in enumerate(self.data[i]):
                samples_dict[j] = row

            [haps_clumped, haps_clumped_count] =clusterHaps(numSamples,samples_dict)

            if ordering == 'rows_freq':
                haps_clumped_count_sort =dict(sorted(haps_clumped_count.items(), key=lambda item: item[1], reverse=True))
                counter = 0
                gene_data_sorted_lst=[]
                for hap in haps_clumped_count_sort.keys():
                    for id in haps_clumped[hap]:
                        gene_data_sorted_lst.append(samples_dict[id])
                        counter +=1
                gene_data_sorted = np.array(gene_data_sorted_lst)
                self.data[i][:,:,:] = gene_data_sorted
            
            elif ordering == 'rows_dist':
                haps_clumped_count_sort =dict(sorted(haps_clumped_count.items(), key=lambda item: item[1], reverse=True))
                #now order by distance to hap of highest freq
                haps_clumped_distance = clusterHaps_byDistance(haps_clumped_count_sort,0.9)
                #sort from least to greatest distance
                haps_clumped_distance_sort = dict(sorted(haps_clumped_distance.items(), key=lambda item: item[1], reverse=False))
                #Now sort data
                counter = 0
                gene_data_sorted_lst=[]
                for hap in haps_clumped_distance_sort.keys():
                    for id in haps_clumped[hap]:
                        gene_data_sorted_lst.append(samples_dict[id])
                        counter +=1
                gene_data_sorted = np.array(gene_data_sorted_lst)
                self.data[i][:,:,:] = gene_data_sorted
            else:
                print('Select a valid ordering.')
                return 1

        return 0

    def convert(self, normalise=False, flip=False, verbose=False):
        """
        Check for correct data type and convert otherwise. Convert to float numpy arrays [0,1] too. If flip true, then flips 0-1
        """
        # if list, put is as numpy array
        if type(self.data) == list:
            if len(np.unique(self.dimensions[0]))*len(np.unique(self.dimensions[1])) == 1:
                if verbose:
                    print('Converting to numpy array.')
                self.data = np.asarray(self.data)
            else:
                print('Aborted. All images must have the same shape.')
                return 1
        # if unit8, put it as float and divide by 255
        if self.data.dtype != 'float32':
            if verbose:
                print('Converting to float32.')
            self.data = self.data.astype('float32')
        if np.nanmax(self.data) > 1:
            if verbose:
                print('Converting to [0,1].')
            self.data /= 255.
        # normalise
        if normalise==True:
            if verbose:
                print('Normalising samplewise.')
            for i in range(len(self.data)):
                mean = self.data[i].mean()
                std = self.data[i].std()
                self.data[i] -= mean
                self.data[i] /= std
        # flip
        if flip==True:
            if verbose:
                print('Flipping values.')
            for i in range(len(self.data)):
                self.data[i] = 1. - self.data[i]
        if verbose:
            if self.data.shape[0] > 1: 
                print('A numpy array with dimensions', self.data.shape, 'and', len(self.targets), 'targets and', len(self.classes), 'classes.')
            else: # one real image
                print('A numpy array with dimensions', self.data.shape)
        return 0

    def set_classes(self, classes=[], nr_classes=0):
        """
        Set classes (or reinitiate)
        """
        # at each call reinitialise for safety
        targets = np.zeros(len(self.data), dtype='int32')
        for i in range(len(self.data)):
            # set target from file description
            targets[i] = self.description[i][self.parameter_name]
        self.classes = np.unique(targets)
        # calculate and/or assign new classes
        if nr_classes > 0:
            self.classes = np.asarray(np.linspace(targets.min(), targets.max(), nr_classes), dtype='int32')
        elif len(classes)>0:
            self.classes = classes
        del targets
        return 0

    def set_targets(self):
        """
        Set targets for binary or categorical classification (not for regression) AFTER running set_classes
        """
        # initialise
        self.targets = np.zeros(len(self.data), dtype='int32')
        for i in range(len(self.targets)):
            # reinitialise
            self.targets[i] = self.description[i][self.parameter_name]
            # assign label as closest class
            self.targets[i] = self.classes[np.argsort(np.abs(self.targets[i] - self.classes))[0]]
        return 0
    def subset_window(self, index):
        """
        get subset window of chromosome to run DANN scan
        """
        sub_data = self.data[0][:,index,:]
        sub_pos = self.positions[0][index]
        sub_gene = ImaGene(data = [sub_data], positions = [sub_pos])
        return sub_gene
    
    def subset(self, index):
        """
        Subset object to index array (for shuffling or only for multiclassification after setting classes and targets)
        """
        # update based on index
        self.targets_classifier = self.targets_classifier[index] 
        self.targets_discriminator = self.targets_discriminator[index]
        self.data = self.data[index]
        self.positions = [self.positions[i] for i in index]
        self.description = [self.description[i] for i in index]
        for i in range(len(self.data)):
            self.dimensions[0][i] = self.data[i].shape[0]
            self.dimensions[1][i] = self.data[i].shape[1]
        return 0

    def save(self, file):
        """
        Save to file
        """
        with open(file, 'wb') as fp:
            pickle.dump(self, fp)
        return 0
    
    def sparse(self):
        for i, image in enumerate(self.data):
            print(i)
            print(image)

    def random_sites_sample(self,new_size):
        """
        In window of legth n SNPS get random subsample of new_size<n snps
        """
        for i, image in enumerate(self.data):
            x, y, c = image.shape[0], image.shape[1], image.shape[2]
            selected_indices = np.random.choice(y, new_size, replace=False)
            selected_indices.sort()

            self.data[i] = image[:, selected_indices, :]

            self.dimensions[0][i] = self.data[i].shape[0]
            self.dimensions[1][i] = self.data[i].shape[1]

        print("New dimensions",self.data[0].shape)
        return None



    def crop(self, window):
        """
        crop or extend haplotype window for genomic image object. Window size are adjusted from center

        Arguments:
            window: haplotype window size


        Note: padding doesn't work well if window size is odd.
        padding gives an even window result

        """

        for i, image in enumerate(self.data):
            
            x, y, c = image.shape[0], image.shape[1], image.shape[2]
            #print(image.shape)
            if y == window:
                continue
            
            #when even no. haplotype column
            if y % 2 == 0:
                if window < y:
                    starty = y // 2 - window // 2
                    self.data[i] = image[:, starty:starty + window, :]

                #perform padding
                else:
                    padding_len = (window - y) // 2
                    padding = np.zeros((x, padding_len, c))
                    if (y+padding_len*2<window): #If it rounds down to nearest even number
                        self.data[i] = np.concatenate((padding, image, padding,np.zeros((x, 1, c))), axis=1)
                    elif (y+padding_len*2>window):
                        padding2 = np.zeros((x, padding_len-1, c)) #If it rounds up to nearest even number
                        self.data[i] = np.concatenate((padding, image, padding2), axis=1)
                    else: # add equal padding to left and right
                        self.data[i] = np.concatenate((padding, image, padding), axis=1)

            
            #when odd no.haplotype column
            #will result in slight offset for window by padding a empty padding on the right hand side
            else:
                offset_padding = np.zeros((x, 1, c))
                image = np.concatenate((image, offset_padding), axis = 1)
                #perform cropping
                if window < y:
                    starty = y // 2 - window // 2
                    self.data[i] = image[:, starty:starty + window, :]

                #perform padding
                else:
                    padding_len = (window - y) // 2
                    padding = np.zeros((x, padding_len, c))
                    if (y+padding_len*2+1 < window): #If it rounds down to nearest even number. The +1 is from the offset padding
                        self.data[i] = np.concatenate((padding, image, padding,np.zeros((x, 1, c))), axis=1)
                    elif (y+padding_len*2+1 > window):
                        l=y+padding_len*2+1-window
                        padding2 = np.zeros((x, padding_len-l, c)) #If it rounds up to nearest even number
                        self.data[i] = np.concatenate((padding, image, padding2), axis=1)
                    else: # add equal padding to left and right
                        self.data[i] = np.concatenate((padding, image, padding), axis=1)
                    #print(self.data[i].shape)

            
            #update dimension
            self.dimensions[0][i] = self.data[i].shape[0]
            self.dimensions[1][i] = self.data[i].shape[1]
            if (self.data[i].shape[1] != window):
                print("WRONG WINDOW SIZE")
                print(self.data[i].shape[1])
                print(i)
        return None



class popGenStats:
    '''
        compute population genetic statistics on .dat data processed with Imagene
    '''
    def __init__(self, file,window_length):

        gene = load_imagene(file) #load .dat processed ImaGene objects

        self.data = gene.data
        self.positions = gene.positions
        self.window_length =window_length # total window length (ex: 450 KB in simulations)

        return None

    #---------------- Utils --------------
    def findClusters(self,haps):
        # This definition identifies haplotypes present in the sample in at least 2 individuals.
        missing_thresh=0.75
        n_min1=2
        n_min2=2

        
        # find the top clusters comprised of at least n_min members
        clusters = {}

        # flag for first cluster > n_min1 found
        n_min_found = False
        
        for key in haps.keys():
                key_split =key.split(',')
                if key_split.count('nan') >= int(len(key_split)*missing_thresh): #if more missing data than threshold, continue
                    continue
                if len(haps[key]) > int(n_min1)-1:
                    clusters[key] = [] # Store the top clusters in this dictionary
                    n_min1_found = True
                    
        if n_min1_found == True:
                for key in haps.keys():
                    if (len(haps[key]) > int(n_min2)-1 and len(haps[key]) < int(n_min1) ):
                        clusters[key] = []
        return clusters


    def sortClusters(self,clusters,haps):
        # this definition sorts haplotype clusters in reverse order from largest to smallest. This sorting will help in the computation of haplotype homozygosity statistics H12, H2, and H1.

        # First order the keys for each cluster from largest to smallest:
        keyVector = []
        sizeVector = []
        for key in clusters.keys():
            keyVector.append(key)
            sizeVector.append(len(haps[key]))
            
        # now sort using bubble sort (need to sort in place):
        swapped = True
        while swapped == True:
            swapped = False
            for i in range(0, len(sizeVector)-1):
                if sizeVector[i] < sizeVector[i+1]:
                    tmpSize = sizeVector[i]
                    sizeVector[i] = sizeVector[i+1]
                    sizeVector[i+1] = tmpSize
                    tmpKey = keyVector[i]
                    keyVector[i] = keyVector[i+1]
                    keyVector[i+1]=tmpKey
                    swapped = True

        return [keyVector, sizeVector]

    # ------------ Statistics ------------------
    def pi(self):
        sample_size=self.data.shape[1]

        pi_per_bp_array = np.zeros(len(self.data), dtype='float32')
        for i in range(len(self.data)):
            ma_counts=np.sum((self.data[i] == 0),axis=0) # total counts of minor allele
            total_counts=np.sum(( ~np.isnan(self.data[i])),axis=0) #total counts of non missing values
            p = ma_counts/total_counts #minor allele frequency per site
            pi_per_site=2*p*(1-p)*sample_size/(sample_size-1) #pi or heterozigosity at each site in the window
            pi_per_bp_array[i]= sum(pi_per_site)/self.window_length #pi/bp in window
            print(pi_per_bp_array[i])
            if(pi_per_bp_array[i]== np.nan):
                print("window_length",self.window_length)
                print("sample size", sample_size-1)
                print("total non-na counts", total_counts)
                print("MA counts",ma_counts)

        return pi_per_bp_array


    def HapStats(self):
        numSamples=self.data.shape[1]

        K_array = np.zeros(len(self.data), dtype='int32') #num haplotypes
        H1_array = np.zeros(len(self.data), dtype='float32')
        H12_array = np.zeros(len(self.data), dtype='float32')
        ratioH2H1_array =np.zeros(len(self.data), dtype='float32')

        for i in range(len(self.data)):
            samples_dict = {} #empty dict


            #create dictionary of samples
            for j, row in enumerate(self.data[i]):
                samples_dict[j] = row
            
            # clump haplotypes that differ by some min threshold 
            [haps_clumped, haps_clumped_count] =clusterHaps(numSamples,samples_dict)

            # find all clusters with at least 2 haplotypes
            clusters = self.findClusters(haps_clumped)

            sizeVector = []
            keyVector = []
            
            if (len(list(clusters.keys())) == 0): #check if I dont find clusters with at least 3 haps
                for key in haps_clumped.keys():
                    sizeVector.append(1)
                    keyVector.append(key)
            else:
                [keyVector, sizeVector] = self.sortClusters(clusters,haps_clumped)

            K, H1, H12, ratioH2H1 = self.computeH12stats(clusters,keyVector,sizeVector,numSamples)

            K_array[i] = K
            H1_array[i]= H1
            H12_array[i]= H12
            ratioH2H1_array[i]= ratioH2H1

        return K_array,H1_array,H12_array,ratioH2H1_array
    
    def computeH12stats(self,clusters,keyVector,sizeVector,numSamples):
        H1 =0
        H2 =0
        H12 = 0
        H123 =0
        ratioH2H1 =0
        
        H1_vector = []
        
        for y in range(0,len(sizeVector)):
            H1 +=  (float(sizeVector[y])/float(numSamples))**2
            H1_vector.append((float(sizeVector[y])/float(numSamples))**2)
        
        # Add on the singletons to H1:
        H1 += (float(numSamples)-sum(sizeVector))*(1/float(numSamples))**2
         
        if len(sizeVector) >0 :
            H2 = H1 - (float(sizeVector[0])/float(numSamples))**2
        else:
            H2 = H1 - (1/float(numSamples))**2
            

        if len(sizeVector) >1 :
            # calculate H12:
            H12 = ((float(sizeVector[0])+float(sizeVector[1]))/float(numSamples))**2
        elif len(sizeVector) ==1:  # If there is only 1 cluster to be found, H12 = H1
            H12 = ((float(sizeVector[0]))/float(numSamples))**2
        else: # if there are zero clusters:
            H12 = (2/float(numSamples))**2 -2*(1/float(numSamples))**2

            
        if len(sizeVector) >2:
            # finish computing H12
            for y in range(2,len(sizeVector)):
                H12 += (float(sizeVector[y])/float(numSamples))**2
            # compute H123:
            H123 = ((float(sizeVector[0])+float(sizeVector[1])+float(sizeVector[2]))/float(numSamples))**2

        # add on the singletons:
        H12+= (float(numSamples)-sum(sizeVector))*(1/float(numSamples))**2

        if len(sizeVector) >3:
            # finish computing H123:
            for y in range(3,len(sizeVector)):
                H123 += (float(sizeVector[y])/float(numSamples))**2

        # add on the singletons
        H123+=(float(numSamples)-sum(sizeVector))*(1/float(numSamples))**2

        if H1 > 0:
            ratioH2H1 = float(H2)/float(H1)

            
        #In the case where there are zero clusters:
        if len(list(clusters.keys())) == 0:
            a = str(0)
            membersOfClusters = '()'

        # print the total number of unique haplotypes found in the sample (K)
        K = len(sizeVector) + numSamples - sum(sizeVector)

    
        return K, H1, H12, ratioH2H1










    
