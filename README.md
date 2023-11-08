# NNMOBILE-NET

**Abstract** 
In the past decades, convolutional neural networks (CNNs) have dominated in diagnosing and monitoring various retinal diseases (RD). However, since the advent of the vision transformer (ViT) in the 2020s, the advancement of recent RD models has been aligned with this trend. The state-of-the-art performance of ViT-based RD models is mainly attributed to their excellent scaling behavior, which implies adding more parameters leads to performance gains. Consequently, although ViT-based models generally (not universally) outperform traditional CNNs in various RD tasks, they are more data-hungry and computationally intense than CNNs. In addition, unlike CNNs, which operate on local regions, ViTs operate at the image patch level, hindering their accurate localization of small lesions with heterogeneous appearance in different RD tasks. In this paper, we revisit the design of a CNN model (i.e., MobileNet) to modernize its performance for RD tasks. We demonstrated that a well-calibrated lightweight MobileNet with several modifications outperformed ViT-based RD models in multiple benchmark tasks, including diabetic retinopathy grading, fundus multi-disease detection, and diabetic macular edema classification.

## Experiments
<img src="image/table1.png"/>

<img src="image/table2.png"/>

<img src="image/table3.png"/>

<img src="image/table4.png"/>


### This repository will be maintained and updated! Stay Tuned!
We will appreciate any suggestions and comments. If you find this code helpful, please cite our papers. Thanks! 
```
@article{zhu2023nnmobile,
  title={NNMobile-Net: Rethinking CNN Design for Deep Learning-Based Retinopathy Research},
  author={Zhu, Wenhui and Qiu, Peijie and Lepore, Natasha and Dumitrascu, Oana M and Wang, Yalin},
  journal={arXiv preprint arXiv:2306.01289},
  year={2023}
} 
```

  ## License

  Released under the [ASU GitHub Project License](https://github.com/Retinotopy-mapping-Research/DRRM/blob/master/LICENSE.txt).
