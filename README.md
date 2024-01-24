# VLExpan
This is the official github repository for the paper "VLExpan: A Visual-Enhanced LLM Framework with Inductive and Deductive Policy for Entity Set Expansion".

We present the source code release the NERD-Img dataset.

## Contents

- [VLExpan](#VLExpan)
  - [Contents](#contents)
  - [Overview](#overview)
  - [Dataset](#Dataset)
  - [Data procession](#Data-procession)
  - [Data Format](#data-format)
  - [Dataset Construction](#dataset-construction)
  - [Downstream Tasks](#downstream-tasks)
  - [Resource Maintenance Plan](#Resource-Maintenance-Plan)
  - [License](#license)

## Overview
<img src="VLExpan.jpg"/>

Our proposed VLExpan consists of four key steps: (1) entity representation, (2) expanding and selecting (3) class name induction and (4) entity deduction.

## Dataset
The entity, courpus, query and ground truth of NERD-Img dataset can be found in "src/data/NERD/". 
Due to the anonymous policy, the all image resource will be released in Google Driver after anonymous restriction. We give some cases in "image-crawler". 

## Data procession
Run the following instruction to tokenize the corpus, which returns 'entity2sents_beit3.pkl'.
```python
>>> pyton -u src/make_entity2sents_beit3.py
```
