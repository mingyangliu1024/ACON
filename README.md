# ACON (NeurIPS 2024)
Official implementation of our NeurIPS 2024 paper [Boosting Transferability and Discriminability for Time Series Domain Adaptation](https://openreview.net/pdf?id=cIBSsXowMr).

## Contributions
- We uncover the characteristics wherein temporal features and frequency features cannot be equally treated in transfer learning. Specifically, we observe that frequency features are more discriminative within a specific domain, while temporal features show better transferability across domains through empirical findings.

- We design ACON, which enhances UDA in three key aspects: a multi-period feature learning module to enhance the discriminability of frequency features, a temporal-frequency domain mutual learning module to enhance the discriminability of temporal features in the source domain and improve the transferability of frequency features in the target domain, and a domain adversarial learning module in temporal-frequency correlation subspace to further enhance transferability of features.

- Experiments conducted on eights time series datasets and five common applications verify the effectiveness.

## Datasets
- [UCIHAR](https://researchdata.ntu.edu.sg/dataset.xhtml?persistentId=doi:10.21979/N9/0SYHTZ)
- [HHAR-P](https://researchdata.ntu.edu.sg/dataset.xhtml?persistentId=doi:10.21979/N9/OWDFXO)
- [WISDM](https://researchdata.ntu.edu.sg/dataset.xhtml?persistentId=doi:10.21979/N9/KJWE5B)
- [FD](https://researchdata.ntu.edu.sg/dataset.xhtml?persistentId=doi:10.21979/N9/PU85XN)
- [HHAR-D](https://woods-benchmarks.github.io/hhar.html)
- [EMG](https://github.com/microsoft/robustlearn/tree/main/diversify)
- [PCL](https://woods-benchmarks.github.io/pcl.html)
- [CAP](https://woods-benchmarks.github.io/cap.html)

The train/test set split of HHAR_D, EMG, PCL, CAP:

- [HHAR_D](https://drive.google.com/file/d/17CBFtmJpxXNyxgA5HH7gB0JURdJdIA04/view?usp=sharing)
- [EMG](https://drive.google.com/file/d/1RgO8gtJSecdqGyir7RamjjqbcesFsFce/view?usp=sharing)
- [PCL](https://drive.google.com/file/d/1MUVL4_cUcNHeKEVafaf-h2uGretcRAfH/view?usp=sharing)
- [CAP]()


Data directory structure
```
.
└── data
    └── CAP
        ├── test_0.pt
        ├── test_1.pt
        ├── test_2.pt
        ├── test_3.pt
        ├── test_4.pt
        ├── train_0.pt
        ├── train_1.pt
        ├── train_2.pt
        ├── train_3.pt
        └── train_4.pt
    
    └── UCIHAR
      ......
    └── WISDM
      ......
```

## How to Run
For each dataset, we select **10** source-target domain pairs. 

Detailed domain pairs are provided in [data_model_configs](https://github.com/mingyangliu1024/ACON/blob/main/configs/data_model_configs.py). 

Each experiment is repeated **5** times with different random seeds.

All bash scripts are provided in [scripts](https://github.com/mingyangliu1024/ACON/tree/main/scripts).

To train a model on UCIHAR dataset:
```
CUDA_VISIBLE_DEVICES=0 python main.py \
 --experiment_description ACON \
 --run_description UCIHAR \
 --da_method ACON \
 --dataset UCIHAR \
 --num_runs 5 \
 --lr 0.01 \
 --cls_trade_off 1 \
 --domain_trade_off 1 \
 --entropy_trade_off 0.01 \
 --align_t_trade_off 1 \
 --align_s_trade_off 1
```

## Acknowledgement
This repo is built on the pioneer works. We appreciate the following GitHub repos a lot for their valuable code base or datasets:

- [RAINCOAT](https://github.com/mims-harvard/Raincoat)

- [AdaTime](https://github.com/emadeldeen24/AdaTime)

## Citation
If you find this work helpful for your research, please kindly cite the following paper:
```
@inproceedings{liuboosting,
  title={Boosting Transferability and Discriminability for Time Series Domain Adaptation},
  author={Liu, Mingyang and Chen, Xinyang and Shu, Yang and Li, Xiucheng and Guan, Weili and Nie, Liqiang},
  booktitle={Annual Conference on Neural Information Processing Systems (NeurIPS)},
  year={2024}
}
```