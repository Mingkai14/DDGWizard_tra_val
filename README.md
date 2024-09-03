# DDGWizard

**Background:**  
Thermostability is an important property of proteins and a critical factor for their wide application. Until now, rational/semi-rational design combined with computational methods have become widely used strategies to increase protein thermostability. ΔΔG prediction methods based on machine learning are among these computational methods and have been widely proposed [1]. However, they still suffer from the issue of insufficient prediction performance [2]. The main reasons include that the features used for training models are insufficiently informative [2]. 

**Characteristics:**  
To conduct more sufficient feature engineering, we constructed a comprehensive ΔΔG feature set by integrating current ΔΔG feature resources and developed a feature extraction pipeline to extract features from raw ΔΔG data. Furthermore, feature dimensionality reduction was conducted to select the optimal features and develop a ΔΔG prediction model. The model showed notable performance, achieving an R-squared of 0.61 in cross-validation and outperformed other representative ΔΔG prediction methods (ACDC-NN[3], DDGun3D[4], FoldX[5], DynaMut[6], DUET[7], mCSM[8], and SDM[9]) in different comparisons. The developed feature extraction pipeline and ΔΔG prediction model constituted our new ΔΔG prediction system, named DDGWizard.

**Purpose:**  
To ensure reproducibility, the training and validation process's source code as well as related data of DDGWizard have been published here.  

# Installation

## Recommendate to use python 3.10.13 to install, which is the python version when developing

## pip install -r requirements.txt

# Reference

[1] Marabotti A, Scafuri B, Facchiano A. Predicting the stability of mutant proteins by computational approaches: an overview[J]. Briefings in Bioinformatics, 2021, 22(3): bbaa074.  
[2] Fang J. A critical review of five machine learning-based algorithms for predicting protein stability changes upon mutation[J]. Briefings in bioinformatics, 2020, 21(4): 1285-1292.  
[3] Benevenuta S, Pancotti C, Fariselli P, et al. An antisymmetric neural network to predict free energy changes in protein variants[J]. Journal of Physics D: Applied Physics, 2021, 54(24): 245403.  
[4] Li B, Yang Y T, Capra J A, et al. Predicting changes in protein thermodynamic stability upon point mutation with deep 3D convolutional neural networks[J]. PLoS computational biology, 2020, 16(11): e1008291.  
[5] Guerois R, Nielsen J E, Serrano L. Predicting changes in the stability of proteins and protein complexes: a study of more than 1000 mutations[J]. Journal of molecular biology, 2002, 320(2): 369-387.  
[6] Rodrigues C H M, Pires D E V, Ascher D B. DynaMut: predicting the impact of mutations on protein conformation, flexibility and stability[J]. Nucleic acids research, 2018, 46(W1): W350-W355.  
[7] Pires D E V, Ascher D B, Blundell T L. DUET: a server for predicting effects of mutations on protein stability using an integrated computational approach[J]. Nucleic acids research, 2014, 42(W1): W314-W319.  
[8] Pires D E V, Ascher D B, Blundell T L. mCSM: predicting the effects of mutations in proteins using graph-based signatures[J]. Bioinformatics, 2014, 30(3): 335-342.  
[9] Pandurangan A P, Ochoa-Montano B, Ascher D B, et al. SDM: a server for predicting effects of mutations on protein stability[J]. Nucleic acids research, 2017, 45(W1): W229-W235.  

