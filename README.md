# MAGNET_MaKE_Experiments

This repository contains the code to perform a number of experiments found in the report "Evaluating Math Word Problem Generation Techniques". All the code here was my own work. The code is split into several subsections.

## Process Generated Datasets 

This section provides code used to process the datasets generated by the MaKE model. Output ".pt" files directly from the MaKE model can be placed in the "generated_datasets" folder and used to run the processing experiments to create a useable dataset for MWP Solver models.

## Process Original Dataset 

Code to process the original MaKE dataset is presented here. The original dataset was in an Excel format, the preprocessing here gets the data in a format read for use with MWP-SS-Metrics or MWP Solver models.

## Similarity Score 

Small programs that were used to run the experiments for cosine similarity ratio and metric values (BLEU, ROUGE, METEOR scores) are presented here. These are used to generate the graph's produced in the MaKE and MAGNET results section of the report as well as other small experiments present in the papers.

## To Use the Scripts 

In order to run the code here, appropriate data for processing or running the experiments should be added to the relavent file names.
