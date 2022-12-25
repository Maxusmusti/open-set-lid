# Open-Set Speech Language Identification and the CU MultiLang Dataset

#### Mustafa Eyceoz, Justin Lee, Siddarth Pittie

## Publication

Robust and Accessible: The CU MultiLang Dataset and Continuing Open-Set Speech LID
 - URL PENDING
 - Please read the paper for a summary of both the new multi-language speech dataset, as well as the goals, architecture, and results of our language identification system.

## Previous Works

Modernizing Open-Set Speech Language Identification
 - Our original paper that inspired this continued work
 - [arXiv Publication URL](https://arxiv.org/abs/2205.10397)
 - [Git Repo](https://github.com/jjlee0802cu/open-set-lid)

## Project Summary
Building an accessible, robust, and general solution for open-set speech language identification.
 - Capable of identifying know languages with high accuracy, but also recognizing and learning unknown languages on-the-fly without having to retrain the foundation TDNN model.
 - Highly portable, with not only full-system inference being possible on incredibly lightweight hardware, but even full model training on reasonable developer hardware (single-gpu, 32GB RAM system can train in a matter of hours).

Also building a diverse, high-coverage, open-source speech dataset spanning over 50 languages.
 - Used to make the system robust and generalized.
 - Not only coverage of most language families, but targeted diversity in speakers and dialects within each language as well.

## CU MultiLang Dataset
The full dataset can be accessed at the below link:
 - [Link to Dataset](https://console.cloud.google.com/storage/browser/cu-multilang-dataset)

## LID System Code Guide

**Run demo**

```
$ python3 full_system.py
```

**Train TDNN with 5hrs, 4sec, 15 epochs**

```
$ python3 train_tdnn.py 5 4 15 0.8
```

**Test the tdnn-final-submission TDNN model**

```
$ python3 test_tdnn.py ./saved-models/tdnn-final-submission.pickle
```

**Save the outputs of tdnn-final-submission TDNN model**

```
$ python3 get_tdnn_outputs.py ./saved-models/tdnn-final-submission.pickle
```

**Train LDA and pLDA layers using saved-tdnn-outputs**

```
$ python3 train_lda_plda.py ./saved-tdnn-outputs
```

## Feature Generation
MFCC + pitch features were generated using Kaldi
 - MFCC and pitch conf files can be found in the `mfcc-confs` subdir
   - Originally found in `kaldi/egs/tedlium/s5_r3/scripts/conf/mfcc.conf` and `kaldi/egs/tedlium/s5_r3/scripts/conf/pitch.conf` 
 - Also in that subdir is our modified version of the `make_mfcc_pitch.sh` script
   - Originally found in and runnable from `kaldi/egs/tedlium/s5_r3/scripts/steps/make_mfcc_pitch.sh`
   - Usage: `make_mfcc_pitch.sh --nj 1 --cmd "$train_cmd" <language directory> <log directory> <mfcc_pitch output directory>` 

## Open-Source Citations
Language Data Sources
 - [VoxForge](http://www.voxforge.org/home)
 - [VoxLingua107](http://bark.phon.ioc.ee/voxlingua107/)
 - [MediaSpeech](https://openslr.org/108/)
 - [BibleTTS](https://openslr.org/129/)
 - [African Accented French](https://openslr.org/57/)
 - [Free ST American English Corpus](https://openslr.org/45/)
 - [FHNW Swiss Parliament](https://huggingface.co/datasets/Yves/fhnw_swiss_parliament)
 - [Samromur 21.05](https://www.openslr.org/112/)
 - [Russian LibriSpeech](https://openslr.org/96/)
 - [Iban](https://www.openslr.org/24/)
 - [THUYG-20](https://www.openslr.org/22/)
 - [Zeroth-Korean](https://www.openslr.org/40/)
 - [Kashmiri Data Corpus](https://www.openslr.org/122/)
 - [Large Bengali ASR training data set](https://openslr.org/53/)
 - [Crowdsourced high-quality Chilean Spanish speech data set](https://www.openslr.org/71/)
 - [Crowdsourced high-quality Colombian Spanish speech data set](https://www.openslr.org/72/)
 - [Crowdsourced high-quality Peruvian Spanish speech data set](https://www.openslr.org/73/)
 - [Crowdsourced high-quality Puerto Rico Spanish speech data set](https://www.openslr.org/74/)
 - [Crowdsourced high-quality Burmese speech data set](https://openslr.org/80/)
 - [Crowdsourced high-quality Telugu multi-speaker speech data set](https://openslr.org/66/)
 - [Crowdsourced high-quality Malayalam multi-speaker speech data set](https://openslr.org/63/)
 - [Crowdsourced high-quality Tamil multi-speaker speech data set](https://openslr.org/65/)
 - [Crowdsourced high-quality Catalan speech data set](https://openslr.org/69/)

Basic PyTorch TDNN Reference
 - https://github.com/cvqluu/TDNN

Python pLDA Reference 
- https://github.com/RaviSoji/plda
