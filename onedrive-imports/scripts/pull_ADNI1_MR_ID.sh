#!/bin/sh

echo here 

for d in /Volumes/cerebro-1/Studies/ADNI_2020/Public/Analysis/data/ADNI1/ADNI1_data/*; 
	do [[ -d "$d" ]] && echo "$d" >> /Volumes/cerebro-1/Workspaces/Students/Anzu_Sekikawa/ADNI1_MR_ID.txt; 
done

echo done 


