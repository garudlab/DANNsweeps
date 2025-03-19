#!/bin/bash
# This script takes in an MS file  with 1000 simulations, and outputs a concatenated processed file. 
file=$1
Q=$2
id=$3

outFile=MSfiles/Neutral_Souilmi_Q${Q}_${id}.txt

for i in `seq 1 1500`; do  #2500

	#get slim command 

	python Souilmi_burn_in_and_slim_command.py ${id} ${i} ${Q} slim > ${file}_var

	command=`cat ${file}_var | head -1`
	echo $command

	#get mu and rho
	rho=$(echo "$command" | sed -n 's/.*recomb_rate=\([^ ]*\).*/\1/p')
	mu=$(echo "$command" | sed -n 's/.*MU=\([^ ]*\).*/\1/p')

	echo "rho: $rho"
	echo "mu: $mu"
	
	#run slim post burn in
	eval $command

    #add mutations and recapitate
    python3.7 add_mutations.py tmp_intermediate_files/Souilmi_${id}${i}.trees ${file} $mu $Q
     	
	echo "" >> ${outFile} 
	cat ${file} >> ${outFile}

	rm tmp_intermediate_files/Souilmi_burn_in_${id}${i}.trees
	rm tmp_intermediate_files/Souilmi_${id}${i}.trees
done

rm ${file}
rm ${file}_var

