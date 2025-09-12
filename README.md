# patent_time_split

We analyze compounds using the SureChEMBL dataset with a focus on time-series data.  
Our study explores time-aware toxicity prediction and the relationship between compound structures and temporal trends.

## Publication
Not yet.  
  
## Organization
------------
    .
    ├── README.md
    ├── data
    │   ├── defreezed
    │   │   ├── drugbank
    │   │   └── surechembl
    │   ├── processed
    │   │   ├── drugbank
    │   │   ├── moleculenet
    │   │   ├── surechembl
    │   │   └── tdc
    │   ├── raw
    │   │   ├── drugbank
    │   │   ├── dsstox
    │   │   └── surechembl
    │   └── result
    │       ├── combined_p_value_cls_2.csv
    │       ├── combined_p_value_cls_3.csv
    │       ├── moleculenet_scores
    │       │   ├── bace_classification_Class_moleculenet_20%.tsv
    │       │   └── bace_classification_Class_moleculenet_30%.tsv
    │       ├── overlap
    │       ├── tdc_scores
    │       │   ├── tdc_20%.tsv
    │       │   ├── tdc_30%.tsv
    │       │   ├── tdc_reg_20%.tsv
    │       │   └── tdc_reg_30%.tsv
    │       └── tmap
    │           ├── 0
    │           ├── scores.csv
    │           └── surechembl
    ├── notebook
    │   ├── experiment
    │   │   ├── 1-dataset_overlap.ipynb
    │   │   ├── 10-feature_wise_selection.ipynb
    │   │   ├── 11-tmap.ipynb
    │   │   ├── 11-tmap_iter.ipynb
    │   │   ├── 12-tmap_toxicity.ipynb
    │   │   ├── 2-tox_pred_tdc_reg.py
    │   │   ├── 3-tox_pred_tdc_cls.py
    │   │   ├── 4-tox_pred_mln_reg.py
    │   │   ├── 5-tox_pred_mln_cls.py
    │   │   ├── 6-tox_pred_result.ipynb
    │   │   ├── 7-preprocess_feature_selection.ipynb
    │   │   ├── 7_time_split_graph_analysis.ipynb
    │   │   ├── 8-.ipynb
    │   │   ├── 8_time_split_graph_analysis_toxpred.ipynb
    │   │   └── 9-prediction_patent_time.ipynb
    │   └── preprocess
    │       ├── 1-preprocess_surechembl.ipynb
    │       ├── 2-preprocess_drugbank.ipynb
    │       ├── 3-preprocess_tdc.py
    │       ├── 4-preprocess_moleculenet.py
    │       ├── 5-preprocess_for_transformervae.py
    │       └── 6-preprocess_for_network.py
    ├── src
    │   ├── __init__.py
    │   ├── chemo_process.py
    │   └── util.py
    └── transformerVAE
        ├── 240907_surechembl_mw
        │   ├── accumulates
        │   │   ├── latent
        │   │   ├── mu
        │   │   └── var
        │   ├── checkpoints
        │   │   └── 364490
        │   ├── config.yaml
        │   ├── log.txt
        │   ├── models
        │   │   └── 364490
        │   └── val_score.csv
        └── data
            ├── featured
            │   ├── herg_karim
            │   └── surechembl
            │       ├── 250225_features_1.npy
            │       └── 250225surechembl_mw2000_1
            └── preprocessed
                ├── 250225surechembl_mw2000_1
                ├── surechembl_ecfp.pkl
                ├── surechembl_mol.pkl
                ├── surechembl_mw.txt
                └── surechembl_mw_part1.txt
------------

## Authors
- [Yohei Ohto](https://github.com/YoheiOhto)  
   - main contributor  
- [Tadahaya Mizuno](https://github.com/tadahayamiz)  
  - correspondence  

## Contact
If you have any questions or comments, please feel free to create an issue on github here, or email us:
- oy826c60[at]gmail.com  
- tadahaya[at]gmail.com  
- tadahaya[at]mol.f.u-tokyo.ac.jp  