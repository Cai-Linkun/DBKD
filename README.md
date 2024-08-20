# DBKD
## Dataset
* [MIMIC-III](https://mimic.mit.edu/): The database comprises detailed clinical information regarding >60 000 stays in ICUs at the Beth Israel Deaconess Medical Center in Boston between 2001 - 2012. MIMIC Code Repository can be obtained from: https://github.com/MIT-LCP/mimic-code.
* BFH dataset: 4,899 Chinese EMRs from ***** Hospital between January 2020 and January 2022. The BFH dataset will be made available from the corresponding author by request.

## Requirement
* Python
* Pytorch
* CUDA
* Scikit-learn
* Numpy
* Scipy
* Pandas
* Matplotlib
* nltk
* wget
* gensim
* requests
* tqdm
* transformers
* BeautifulSoup
* tensorboardX
* yaml

## Baselines
* [CAML&DRCAML](https://github.com/jamesmullenbach/caml-mimic)
* [HyperCore](https://aclanthology.org/2020.acl-main.282/)
* [MultiResCNN](https://github.com/foxlf823/Multi-Filter-Residual-Convolutional-Neural-Network)
* [MSATT-KG](https://dl.acm.org/doi/abs/10.1145/3357384.3357897)
* [JAN](https://ieeexplore.ieee.org/document/9822203)

## Label knowledge acquirement
We crawled the ICD description and hierarchy information from [ICD9 website](http://www.icd9data.com/). After adding the parent label, the number of labels in MIMIC-III-full for training, development, and testing are 10,051, 3,928, 5,111, respectively.


## Data usage
Put the files of MIMIC III into the 'data' dir as below:
```
ori_mimic3_data
├── D_ICD_DIAGNOSES.csv
├── D_ICD_PROCEDURES.csv
├── ICD9_descriptions
├── ICD9_descriptions copy
└── mimic3
    ├── DIAGNOSES_ICD.csv
    ├── NOTEEVENTS.csv
    ├── PROCEDURES_ICD.csv
    ├── dev_50_hadm_ids.csv
    ├── dev_full_hadm_ids.csv
    ├── test_50_hadm_ids.csv
    ├── test_full_hadm_ids.csv
    ├── train_50_hadm_ids.csv
    └── train_full_hadm_ids.csv
```

```
processed_data
├── c2ind.npy
├── codes_filled.json
├── comatrix.npy
├── description_vectors.vocab
├── description_words.vocab
├── dev_*.csv
├── disch_dev_split.csv
├── disch_full.csv
├── disch_test_split.csv
├── disch_train_split.csv
├── hier_level_idx.npy
├── hmidx.npy
├── ind2c.npy
├── label_embeddings.pth
├── label_hierarchy.csv
├── notes_labeled.csv
├── processed_full.embed
├── processed_full.w2v
├── test_*.csv
├── train_*.csv
└── vocab.csv
```
## Experiments procedure
## Citation
If you found this work useful for you, please consider citing it.
```
@article{DBKD,
  title   = {DBKD},
  author  = {###},
  journal = {####},
  year    = {####}
}
```
## Contact
For any issues/questions regarding the paper or reproducing the results, please contact any of the following.

##: ###

Department of *****, ****** University
