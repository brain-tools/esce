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
date: 31 May 2023
bibliography: paper.bib
---


# Statement of Need

In recent years, technological advances in neuroimaging have provided unprecedented insight into the structure and function of the human brain. Concurrently, high-resolution brain scans offers hope for automated disease diagnoses and clinical endpoint prediction based on neuroimaging data. Significant progress has been made in some application in fields such as automated detection of brain atrophy and segmentation of brain lesions; while the prediction of cognitive and behavioral phenotypes and the diagnosis of psychiatric diseases has remained challenging [@Kamnitsas2017; @Plant2010; @Woo2017].

Whether such challenges in neuroimaging-based phenotype prediction are primarily due to insufficient sample sizes or a lack of predictive information in neuroimaging data remains to be clarified. The two questions of sample size and exploitable predictive information are closely related. If there was insufficient predictive information in the data, then adding more participants would not improve prediction accuracy. Conversely, if increasing the sample size yields continuous improvements in predictive accuracy, then we hare yet to exhaust the predictive information contained in the data, so there is hope for accurate single subject prediction. If we can mathematically characterize the empirical relationship between sample size and achievable prediction accuracy (or the “learning curve”) for a given target phenotype, it would provide estimates of the necessary sample size to reach a certain accuracy level, as well as estimates of the highest achievable accuracy given infinite samples. 

While the initial need for this workflow roots from experiments in the  field, similar problems can arise when applying machine learning methods to other fields. Consequently, the workflow can be applied to other use cases for calculating necessary sample size and highest achievable accuracy.


## Summary

For efficient calculation of learning curves, we introduce our open-source package ``Empirical Sample Complexity Estimator (esce)`` in this paper. ``Esce`` is a Snakemake workflow that makes it easy to compare the scaling behaviour of various machine learning models, feature sets, and target variables by analyzing the performance of machine learning models as the sample size increases. 

As collecting large neuroimaging datasets is an elaborate and expensive endeavor, we turned to the theoretical results from statistical learning theory, which state that the prediction accuracy typically scales as a power law function of the sample size [@Amari1993; @Amari1992; @aussler1996; @Hutter2021]. Hence, estimating power law parameters from an empirical learning curve allows us to firstly, extrapolate the learning curve beyond the available sample size; and secondly, to infer both the maximally achievable prediction accuracy (from the convergence point of the learning curve) and the necessary sample size to achieve clinically useful performance (from the speed of convergence). Thus quantitatively engaging two core aspects of feasibility for precision medicine. 

The workflow works as follows: for each combination of feature set, target variable, and covariate, it removes rows with missing or NaN values and creates train/validation/test splits for each sample size and random seed defined in the configuration file. It then fits and evaluates models with a range of hyperparameters and saves the results. Afterwards, it collects the accuracy estimates for the best-performing hyperparameter configurations, and uses them to fit power laws to the data, while bootstrapping to estimate uncertainties. Finally, it creates summary figures for analysis. The package also comes with an example workflow to demonstrate how it works and the the results it produces for a better understanding of the tool and provide insights to possible applications.

# Acknowledgments

We thank ... for his/her/their comments and contributions to the project.


# References
