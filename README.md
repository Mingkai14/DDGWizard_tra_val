# Introduction
<p align="justify"">
Thermostability is an important property of proteins and increasing protein thermostability holds significant meaning in expanding the application scope of proteins. Until now, rational/semi-rational design combined with computational methods have become the widely used strategies to increase protein thermostability. Among combined computational methods, the ΔΔG prediction based on machine learning is popular[1], however, optimizing their insufficient accuracy remains a challenge[2]. To achieve accurate ΔΔG prediction, DDGWizard has been developed. DDGWizard introduced a comprehensive feature set potentially related to protein thermostability changes, including 1547 features from 5 types; DDGWizard incorporates a workflow to extract features from raw ΔΔG data, supporting data process in large scale; DDGWizard carried out feature dimensionality reduction to find best feature combination; DDGWizard includes a machine learning model to predict ΔΔG, trained with best features and newly integrated dataset. After model evaluation of DDGWizard and comparison with other representative ΔΔG predictive methods, DDGWizard acheived 0.79 Pearson correlation coefficient in cross-validation and 0.68 Pearson correlation coefficient on test set, outperforming other 7 ΔΔG predictive methods: ACDC-NN[3], DDGun3D[4], FoldX[5], DynaMut[6], DUET[7], mCSM[8], and SDM[9]. The data, training process and validation process have been published here to guarantee the reproducibility.
</p>
#Reference

[1] Marabotti A, Scafuri B, Facchiano A. Predicting the stability of mutant proteins by computational approaches: an overview[J]. Briefings in Bioinformatics, 2021, 22(3): bbaa074.  
[2] Fang J. A critical review of five machine learning-based algorithms for predicting protein stability changes upon mutation[J]. Briefings in bioinformatics, 2020, 21(4): 1285-1292.  
[3] Benevenuta S, Pancotti C, Fariselli P, et al. An antisymmetric neural network to predict free energy changes in protein variants[J]. Journal of Physics D: Applied Physics, 2021, 54(24): 245403.  
[4] Li B, Yang Y T, Capra J A, et al. Predicting changes in protein thermodynamic stability upon point mutation with deep 3D convolutional neural networks[J]. PLoS computational biology, 2020, 16(11): e1008291.  
[5] Guerois R, Nielsen J E, Serrano L. Predicting changes in the stability of proteins and protein complexes: a study of more than 1000 mutations[J]. Journal of molecular biology, 2002, 320(2): 369-387.  
[6] Rodrigues C H M, Pires D E V, Ascher D B. DynaMut: predicting the impact of mutations on protein conformation, flexibility and stability[J]. Nucleic acids research, 2018, 46(W1): W350-W355.  
[7] Pires D E V, Ascher D B, Blundell T L. DUET: a server for predicting effects of mutations on protein stability using an integrated computational approach[J]. Nucleic acids research, 2014, 42(W1): W314-W319.  
[8] Pires D E V, Ascher D B, Blundell T L. mCSM: predicting the effects of mutations in proteins using graph-based signatures[J]. Bioinformatics, 2014, 30(3): 335-342.  
[9] Pandurangan A P, Ochoa-Montano B, Ascher D B, et al. SDM: a server for predicting effects of mutations on protein stability[J]. Nucleic acids research, 2017, 45(W1): W229-W235.  

# DDGWizard_tra_val

This repo includes reproducible programs when DDGWizard training, evaluating, and comparing with other ΔΔG predictive methods, as well as reusable data.

# Installation (using python 3.10.13 when developing)

## pip install -r requirements.txt

# Description of content:
1. folder "1_Datasets": program for process raw dataset. 
2. folder "2_Fea_Groups_Experiment": program for testing performance of different feature tpypes and their combination.  
3. folder "3_Training_Validation": program for carrying out feature engineering, model training and cross-validation.  
4. folder "4_Benchmark": program for benchmarking 7 other representative ΔΔG predictive methods on test set.  
5. folder "5_Test_on_p53: - program for benchmarking 7 other representative ΔΔG predictive methods on p53 dataset.  
