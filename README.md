# Information Theoretic Quantification of Uncertainty in Ensemble Climate Models
Companion notebooks and scripts written to produce results in the paper.

The data used are summarized with the following table from the original thesis:
![image](https://github.com/user-attachments/assets/427d072b-902b-41ce-8964-0e8c75def23c)

When extrapolating datasets, I have chosen to fill unrecorded data with NaN. The calculations are then done either by ignoring them or replacing them with zeros. 

`notebooks/gaussian_arrays.ipynb` contains code used to study the properties of the information theory predictability measures on randomly generated Gaussian samples. 
`noteboks/jugaad.ipynb` investigates different questions regarding how the information theory measures change with respect to the size of the ensemble (using the "tiling" jugaad)
`notebooks/calc_RPC.ipynb` contains the main calculations of the RPC using both variance-based and information-based measures. 

`scripts/DataInventory.py` is a utility script used to keep track of all data files downloaded using `notebooks/download_data.ipynb`. 
`scripts/InfoTheoryMetrics.py` contains functions that calculate entropy and mutual information. 

