# ACON (NeurIPS 2024)
Official implementation of the paper [[Boosting Transferability and Discriminability for Time Series Domain Adaptation](https://openreview.net/pdf?id=cIBSsXowMr)].

## Contributions
- We uncover the characteristics wherein temporal features and frequency features cannot be equally treated in transfer learning. Specifically, we observe that frequency features are more discriminative within a specific domain, while temporal features show better transferability across domains through empirical findings.

- We design ACON, which enhances UDA in three key aspects: a multi-period feature learning module to enhance the discriminability of frequency features, a temporal-frequency domain mutual learning module to enhance the discriminability of temporal features in the source domain and improve the transferability of frequency features in the target domain, and a domain adversarial learning module in temporal-frequency correlation subspace to further enhance transferability of features.

- Experiments conducted on eights time series datasets and five common applications verify the effectiveness of ACON.

## Datasets
- [[UCIHAR](https://researchdata.ntu.edu.sg/dataset.xhtml?persistentId=doi:10.21979/N9/0SYHTZ)]
- [[HHAR-P](https://researchdata.ntu.edu.sg/dataset.xhtml?persistentId=doi:10.21979/N9/OWDFXO)]
- [[WISDM](https://researchdata.ntu.edu.sg/dataset.xhtml?persistentId=doi:10.21979/N9/KJWE5B)]
- [[HHAR-D](https://woods-benchmarks.github.io/hhar.html)]
- [[EMG](https://github.com/microsoft/robustlearn/tree/main/diversify)]
- [[FD](https://researchdata.ntu.edu.sg/dataset.xhtml?persistentId=doi:10.21979/N9/PU85XN)]
- [[PCL](https://woods-benchmarks.github.io/pcl.html)]
- [[CAP](https://woods-benchmarks.github.io/cap.html)]

## How to Run


## Citation
If you find this work helpful for your research, please kindly cite the following paper:
```
@inproceedings{liuboosting,
  title={Boosting Transferability and Discriminability for Time Series Domain Adaptation},
  author={Liu, Mingyang and Chen, Xinyang and Shu, Yang and Li, Xiucheng and Guan, Weili and Nie, Liqiang},
  booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems (NeurIPS)}
}
```