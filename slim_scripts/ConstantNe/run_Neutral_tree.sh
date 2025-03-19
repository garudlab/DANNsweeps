#!/bin/bash
# This script takes in an MS file  with 1000 simulations, and outputs a concatenated processed file. 
file=$1
Ne=$2
Q=$3
chr=$4
id=$5

sexRatio=0.5

outFile=MSfiles/Neutral_${Ne}_sb0_Q${Q}_${id}.txt

for i in `seq 1 4000`; do #2500

	python slim_parametersNeutral.py ${Ne} ${id} ${i} ${Q} slim > ${file}_var

	command=`cat ${file}_var | head -1`
        echo $command

        #get mu and rho
        rho=$(echo "$command" | sed -n 's/.*R=\([^ ]*\).*/\1/p')
        mu=$(echo "$command" | sed -n 's/.*MU=\([^ ]*\).*/\1/p')

        echo "rho: $rho"
        echo "mu: $mu"

        #run slim
        eval $command

	#add mutations and recapitate
    	python3.7 mutate_and_recapitate.py tmp_intermediate_files/ConstantNe_Neutral_${id}${i}.trees ${file} $rho $Ne $mu $Q 


	echo "" >> ${outFile}
        cat ${file} >> ${outFile}     

	rm tmp_intermediate_files/ConstantNe_Neutral_${id}${i}.trees
done
rm ${file}
rm ${file}_var
