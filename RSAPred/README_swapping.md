# Swapping addition 
This is to change the RNA - small molecules pairing in order to assess whether the model is learning from both the RNA and the small molecule or just from the small molecule.

In order to do so, the python file swap_v2.py enables the swapping 
swap_v2.py takes 2 arguments :
- dataset_raw that has to be the negative or positive dataset
- output file name 

The goal is to produce a dataset where:
- each original pair small molecule - RNA is changed to the same small molecule and another RNA (always different, no risk of changing it to the same)
- the pKd and the molecular features are the same as before, the only thing that changes is the RNA
- the dataset structure in the ouput is the same as before, enabling the use of the old python files for the classification tests 

