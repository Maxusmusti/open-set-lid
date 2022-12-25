# Open-Set Speech Language Identification and the CU MultiLang Dataset

#### Mustafa Eyceoz, Justin Lee, Siddarth Pittie

## Publication

TITLE PENDING
 - URL PENDING

## Previous Works

Modernizing Open-Set Speech language identification
 - Our original paper that inspired this continued work
 - [arXiv Publication URL](https://arxiv.org/abs/2205.10397)
 - [Git Repo](https://github.com/jjlee0802cu/open-set-lid)

## Project Summary
Building an accessible, robust, and general solution for open-set speech language identification.
Also building a diverse, high-coverage, open-source speech dataset spanning over 50 languages.

## CU MultiLang Dataset
The full dataset can be accessed at the below link:
 - [Link to Dataset](https://console.cloud.google.com/storage/browser/cu-multilang-dataset)

## LID System Code Guide

Run demo:
 - `python3 full_system.py`

Train TDNN with 5hrs, 4sec, 15 epochs
 - `python3 train_tdnn.py 5 4 15 0.8`

Test the tdnn-final-submission TDNN model
 - `python3 test_tdnn.py ./saved-models/tdnn-final-submission.pickle`

Train lda and plda layers using saved-tdnn-outputs
 - `python3 train_lda_plda.py ./saved-tdnn-outputs`
