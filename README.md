
## LoReTrack

<p align="center"><img src="./images/SPGroup3D.png" alt="drawing" width="90%"/></p>

This project provides the code and results for [LoReTrack: Efficient and Accurate Low-Resolution Transformer Tracking](https://arxiv.org/pdf/2405.17660), IROS 2025 (**Oral**)

Anchors: [Shaohua Dong](https://scholar.google.com/citations?user=5iSEcFkAAAAJ&hl=en), [Yunhe Feng](https://yunhefeng.me/), [James C. Liang](https://jamesliang819.github.io/), [Qing Yang](https://scholar.google.com/citations?user=FIMxNL0AAAAJ&hl=zh-CN), [Yuewei Lin](https://ywlincq.github.io/), [Heng Fan](https://hengfan2010.github.io/)

PaperLink: https://arxiv.org/pdf/2405.17660


## Results and Weights

| Tracker       | LaSOT (AUC)                                                 | LaSOText (AUC)                                              | GOT-10k (AO)                                                 | TrackingNet (AUC)                                              | UAV123 (AUC)                                                |   Device \| FPS    |
|:-------------:|:------------------------------------------------------------:|:------------------------------------------------------------:|:-------------------------------------------------------------:|:--------------------------------------------------------------:|:------------------------------------------------------------:|:------:|
| LoReTrack-256 | 70.3 \| [weight](https://drive.google.com/ost384_weight) \| [raw results](https://drive.google.com/ost384_results) | 51.3 \| [weight](https://drive.google.com/ost384_weight) \| [raw results](https://drive.google.com/ost384_results) | 73.5 \| [weight](https://drive.google.com/ost384_weight) \| [raw results](https://drive.google.com/ost384_results) | 82.9 \| [weight](https://drive.google.com/ost384_weight) \| [raw results](https://drive.google.com/ost384_results) | 70.6 \| [weight](https://drive.google.com/ost384_weight) \| [raw results](https://drive.google.com/ost384_results) | GPU \| 130 |
| LoReTrack-192 | 68.6 \| [weight](https://drive.google.com/ost256_weight) \| [raw results](https://drive.google.com/ost256_results) | 50.0 \| [weight](https://drive.google.com/ost256_weight) \| [raw results](https://drive.google.com/ost256_results) | 71.5 \| [weight](https://drive.google.com/ost256_weight) \| [raw results](https://drive.google.com/ost256_results) | 80.9 \| [weight](https://drive.google.com/ost256_weight) \| [raw results](https://drive.google.com/ost256_results) | 69.9 \| [weight](https://drive.google.com/ost256_weight) \| [raw results](https://drive.google.com/ost256_results) | GPU / 186 |
| LoReTrack-128 | XX.X \| [weight](#) \| [raw results](#)                     | XX.X \| [weight](#) \| [raw results](#)                     | XX.X \| [weight](#) \| [raw results](#)                      | XX.X \| [weight](#) \| [raw results](#)                      | XX.X \| [weight](#) \| [raw results](#)                     | XX FPS |
| LoReTrack-96  | XX.X \| [weight](#) \| [raw results](#)                     | XX.X \| [weight](#) \| [raw results](#)                     | XX.X \| [weight](#) \| [raw results](#)                      | XX.X \| [weight](#) \| [raw results](#)                      | XX.X \| [weight](#) \| [raw results](#)                     | XX FPS |
