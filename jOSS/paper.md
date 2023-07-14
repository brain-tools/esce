---
title: 'Empirical Sample Complexity Estimator: a workflow for calculating necessary sample size and highest achievable accuracy'
tags:
  - machine-learning
  - python
authors:
  - name: Marc-Andre Schulz
    orcid: 
    affiliation: "1, 2"
  - name: Kerstin Ritter
    orcid: 
    affiliation: "1, 2"
  - name: Alexander Koch
    orcid: 
    affiliation: 1
  - name: Chung-Fan Tsai
    orcid: 0009-0009-6018-3367
    affiliation: 1
affiliations:
 - name: Charité – Universitätsmedizin Berlin (corporate member of Freie Universität Berlin, Humboldt-Universität zu Berlin, and Berlin Institute of Health), Department of Psychiatry and Psychotherapy, Berlin, Germany
   index: 1
 - name: Bernstein Center for Computational Neuroscience, Berlin, Germany
   index: 2
date: 7 July 2023
bibliography: paper.bib
---

In this paper, we introduce our open-source package ``Empirical Sample Complexity Estimator (ESCE)``, a Snakemake workflow that makes it easy to compare the scaling behavior of various machine learning models, feature sets, and target variables by analyzing their performance as the sample size increases. 

While assessing the effect of sample size on neuroimaging prediction performance in the paper “Performance reserves in brain-imaging-based phenotype prediction” [@schulz2022], we set out to mathematically estimate the empirical relation between sample size and achievable prediction accuracy. As collecting neuroimaging datasets is elaborate and expensive, we turned to the statistical learning theory, which suggests that learning curve follows a power law function of the sample size [@Amari1993; @Amari1992; @Haussler1996; @Hutter2021]. To further prove the validity of the theory, in the previously mentioned paper, Schulz and colleagues calculated the average goodness-of-fit statistics and observed a comparable result to the power law, showing that the prediction accuracy scales can in fact be closely estimated with the power law function of the increase of sample size. 

The workflow works in the following steps: First, for each combination of feature set, target variable, and covariate, it removes rows with missing or NaN values, then it creates train/validation/test splits for each sample size and defines the random seed in the configuration file. Next, it fits and evaluates models with a range of hyperparameters and saves the results. In the meantime, it collects the accuracy estimates for the best-performing hyperparameter combinations, and uses them to fit power laws to the data, while bootstrapping to estimate uncertainties. Finally, it creates summary figures for analysis. 

The package comes with an example workflow to demonstrate how it works and the results it produces for a better understanding of the tool, along with the documentation, we hope to provide insights  for future research to possible applications in various fields of studies. 


# References