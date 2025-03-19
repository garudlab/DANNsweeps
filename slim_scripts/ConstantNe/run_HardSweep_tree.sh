#!/bin/bash
# This script takes in an MS file  with 1000 simulations, and outputs a concatenated processed file. 
file=$1
Ne=$2
Q=$3
id=$4

outFile=MSfiles/s0.01_0.1/HardSweep_${Ne}_Q${Q}_450Kb_${id}.txt

for i in `seq 1 4000`; do #500, 10000

	#get slim command 

	python slim_parametersHardSweep.py ${Ne} ${id} ${i} ${Q} slim > ${file}_var

	command=`cat ${file}_var | head -1`
	#get mu and rho
        rho=$(echo "$command" | sed -n 's/.*R=\([^ ]*\).*/\1/p')
        mu=$(echo "$command" | sed -n 's/.*MU=\([^ ]*\).*/\1/p')

        echo "rho: $rho"
        echo "mu: $mu"

	echo $command
	eval $command

        #add mutations and recapitate
        python3.7 mutate_and_recapitate.py tmp_intermediate_files/ConstantNe_HS_${id}${i}.trees ${file} $rho $Ne $mu $Q
     	
	echo "" >> ${outFile} 
	cat ${file} >> ${outFile}
	
	rm tmp_intermediate_files/ConstantNe_HS_${id}${i}.trees
done

rm ${file}
rm ${file}_var
