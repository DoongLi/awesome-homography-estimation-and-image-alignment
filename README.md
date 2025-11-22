# awesome-homography-estimation-and-image-alignment[![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)

This repo contains a curative list of **homography estimation, image alignment and related application**. 

> Note: It is worth noting that this repository only focuses on homography estimation methods used for image alignment tasks. Methods for solving poses using homography estimation, such as those on the HPatches benchmarks, are not covered in this repository.

#### Please feel free to send me [pull requests](https://github.com/DoongLi/awesome-homography-estimation-and-image-alignment/blob/main/how-to-PR.md) or [email](mailto:lidong8421bcd@gmail.com) to add papers! <br>

If you find this repository useful, please consider STARing this list. Feel free to share this list with others!

There are some similar repositories available, but it appears that they have not been maintained for a long time.

- https://github.com/tzxiang/awesome-image-alignment-and-stitching
- https://github.com/visionxiang/awesome-computational-photography

## Overview

- [Awesome Homography Estimation and Image Alignment](#awesome-homography-estimation-and-image-alignment)
    - [Homography Estimation](#homography-estimation)
    - [Image Alignment](#image-alignment)
    - [Application](#application)

## Homography Estimation

#### Survey Paper

- A Review of Homography Estimation: Advances and Challenges, `Electronics`, *2023*. [[Paper](https://www.mdpi.com/2079-9292/12/24/4977)]
- Research on Homography Estimation Method Based on Deep Learning, `International Conference on Computational & Experimental Engineering and Sciences`, *2024*. [[Paper](https://link.springer.com/chapter/10.1007/978-3-031-68775-4_46)]

#### 2025

- ![paper](https://img.shields.io/badge/Dataset-red) UASTHN: Uncertainty-Aware Deep Homography Estimation for UAV Satellite-Thermal Geo-localization, `ICRA`. [[Paper](https://arxiv.org/pdf/2502.01035)] [[Code](https://github.com/arplaboratory/UASTHN)] [[Website](https://xjh19971.github.io/UASTHN/)]
- ![paper](https://img.shields.io/badge/Dataset-red) ![page](https://img.shields.io/badge/Pretrain-model-blue) CodingHomo: Bootstrapping Deep Homography with Video Coding, `TCSVT`. [[Paper](https://arxiv.org/pdf/2504.12165)] [[Code](https://github.com/liuyike422/CodingHomo)]
- An Iterative Deep Homography Network Based on Correlation Content Calculation for High-Precision Image Registration of Fly-Capture Imaging System, `TIM`. [[Paper](https://ieeexplore.ieee.org/abstract/document/11030743)]
- Robust Image Stitching with Optimal Plane, `arXiv`. [[Paper](https://arxiv.org/pdf/2508.05903)] [[Code](https://github.com/MmelodYy/RopStitch)]
- Uncertainty-adaptive Volume for Unsupervised Homography Estimation, `TCSVT`. [[Paper](https://ieeexplore.ieee.org/abstract/document/11201867)]
- Image Stitching in Adverse Condition: A Bidirectional-Consistency Learning Framework and Benchmark, `NIPS`. [[Paper](https://openreview.net/pdf?id=l1n22nHG4A)] [[Code](https://github.com/ZengxiZhang/ACDIS)]
- Aerial Image Stitching Using IMU Data from a UAV, `arXiv`. [[Paper](https://arxiv.org/pdf/2511.06841)]
- UAV image stitching method based on dual feature guidance and optimal seam, `Knowledge-Based Systems`. [[Paper](https://www.sciencedirect.com/science/article/abs/pii/S095070512501929X)]

#### 2024

- ![paper](https://img.shields.io/badge/Dataset-red) ![page](https://img.shields.io/badge/Pretrain-model-blue) DMHomo: Learning Homography with Diffusion Models, `ToG`. [[Paper](https://dl.acm.org/doi/full/10.1145/3652207)] [[Code](https://github.com/lhaippp/DMHomo)]
  - keyword: realistic dataset generation, supervised homography estimation
- ![paper](https://img.shields.io/badge/Dataset-red) UMAD: University of Macau Anomaly Detection Benchmark Dataset, `IROS`. [[Paper](https://arxiv.org/pdf/2408.12527)] [[Code](https://github.com/IMRL/UMAD)]
- ![page](https://img.shields.io/badge/Pretrain-model-blue) CrossHomo: Cross-Modality and Cross-Resolution Homography Estimation, `TPAMI`. [[Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10438073&casa_token=92CU3CFbsXIAAAAA:DJN1QCKXfj7pFYlSD4sLh33HUtBQ-BjhiBBRtBtUbcEHjKnSyduvywgyYvAnI9UJW7MkOboG&tag=1)] [[Code](https://github.com/lep990816/CrossHomo)]
  - keyword: cross-modality images, cross-resolution images
- ![page](https://img.shields.io/badge/Pretrain-model-blue) Gyroflow+: Gyroscope-guided unsupervised deep homography and optical flow learning, `IJCV`. [[Paper](https://link.springer.com/article/10.1007/s11263-023-01978-5)] [[Code](https://github.com/lhaippp/GyroFlowPlus)]
- Unsupervised Homography Estimation with Pixel-level SVDD, `TCSVT`. [[Paper](https://ieeexplore.ieee.org/abstract/document/10510339)]
- AbHE: All Attention-Based Homography Estimation, `TIM`. [[Paper](https://ieeexplore.ieee.org/abstract/document/10472928)] [[Code](https://github.com/mingxiaohuo/ABHE)]
- ![paper](https://img.shields.io/badge/Dataset-red) Analyzing the Domain Shift Immunity of Deep Homography Estimation, `WACV`. [[Paper]](https://openaccess.thecvf.com/content/WACV2024/papers/Shao_Analyzing_the_Domain_Shift_Immunity_of_Deep_Homography_Estimation_WACV_2024_paper.pdf)] [[Code](https://github.com/MingzhenShao/Homography_estimation)]
- MCNet: Rethinking the Core Ingredients for  Accurate and Efficient Homography Estimation, `CVPR`. [[Paper](https://github.com/zjuzhk/MCNet/blob/main/CVPR2024-MCNet.pdf)] [[Code](https://github.com/zjuzhk/MCNet)]
- SCPNet: Unsupervised Cross-modal Homography Estimation via Intra-modal Self-supervised Learning, `ECCV`. [[Paper](https://arxiv.org/pdf/2407.08148)] [[Code](https://github.com/RM-Zhang/SCPNet)]
- Deep Homography Estimation via Dense Scene Matching, `RAL`. [[Paper](https://ieeexplore.ieee.org/document/10592770/)]
- STHN: Deep Homography Estimation for UAV Thermal Geo-localization with Satellite Imagery, `RAL`. [[Paper](https://arxiv.org/pdf/2405.20470)] [[Project](https://xjh19971.github.io/STHN/)]  [[Code](https://github.com/arplaboratory/STHN)]
- Implicit Neural Image Stitching With Enhanced and Blended Feature Reconstruction, `WACV`. [[Paper](https://openaccess.thecvf.com/content/WACV2024/papers/Kim_Implicit_Neural_Image_Stitching_With_Enhanced_and_Blended_Feature_Reconstruction_WACV_2024_paper.pdf)] [[Code](https://github.com/minshu-kim/Neural-Image-Stitching)]
- Deep Unsupervised Homography Estimation for Single-Resolution Infrared and Visible Images Using GNN, `Electronics`, *2024*. [[Paper](https://www.mdpi.com/2079-9292/13/21/4173)]
- GFNet: Homography Estimation via Grid Flow Regression, `arXiv`. [[Paper](https://openreview.net/pdf?id=DsW4boRh8H)]
- Homography Estimation with Adaptive Query Transformer and Gated Interaction Module, `TCSVT`. [[Paper](https://ieeexplore.ieee.org/abstract/document/10758195)]
- A method of UAV image homography matrix estimation based on deep learning, `ICDIP`. [[Paper](https://www.spiedigitallibrary.org/conference-proceedings-of-spie/13274/132740N/A-method-of-UAV-image-homography-matrix-estimation-based-on/10.1117/12.3037062.short)]
- Image Registration Algorithm for Stamping Process Monitoring Based on Improved Unsupervised Homography Estimation, `Applied Sciences`. [[Paper](https://www.mdpi.com/2076-3417/14/17/7721)]
- **HomoMatcher**: Dense Feature Matching Results with Semi-Dense Efficiency by Homography Estimation, `arXiv`. [[Paper](https://arxiv.org/pdf/2411.06700)]
- ![paper](https://img.shields.io/badge/Dataset-red) ![page](https://img.shields.io/badge/Pretrain-model-blue) CodingHomo: Bootstrapping Deep Homography With Video Coding, `TCSVT`. [[Paper](https://ieeexplore.ieee.org/abstract/document/10570492)] [[Code](https://github.com/liuyike422/CodingHomo)]
- Unsupervised Global and Local Homography Estimation with Coplanarity-Aware GAN, `TPAMI`. [[Paper](https://ieeexplore.ieee.org/abstract/document/10772056)] [[Code](https://github.com/megvii-research/HomoGAN)]
- SeFENet: Robust Deep Homography Estimation via Semantic-Driven Feature Enhancement, `arXiv`. [[Paper](https://arxiv.org/pdf/2412.06352)]
- Uncertainty Guided Deep Lucas-Kanade Homography for Multimodal Image Alignment, `TGRS`. [[Paper](https://ieeexplore.ieee.org/abstract/document/10816136/authors#authors)]
- Semantic-aware Representation Learning for Homography Estimation, `MM`. [[Paper](https://arxiv.org/pdf/2407.13284)] [[Code](https://github.com/lyh200095/SRMatcher)]

#### 2023

- Edge-aware Correlation Learning for Unsupervised Progressive Homography Estimation, `TCSVT`. [[Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10314523&casa_token=86gKd0-dQBUAAAAA:VJbu9XBitO-BQPYhu2uwW0FI5EMaH2YshfkBnuoHC4jtkCWN8jFEFm2A2Y_Bd5-DU0SChWJP)]
  - keyword: unsupervised homography estimation
- ![page](https://img.shields.io/badge/Pretrain-model-blue) Unsupervised Global and Local Homography Estimation with Motion Basis Learning, `TPAMI`. [[Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9956874&casa_token=-Tp9JpW764gAAAAA:NkiBmmnJ3WHrxXjm5DeEFFXna8haxfiG52I5pWptORdbvSeU2vVZtDl8Y95jbkiIJYnSk2ql)] [[Code](https://github.com/megvii-research/BasesHomo)]
  - keyword: unsupervised homography estimation
- ![paper](https://img.shields.io/badge/Dataset-red) ![page](https://img.shields.io/badge/Pretrain-model-blue) Supervised Homography Learning with Realistic Dataset Generation, `ICCV`. [[Paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Jiang_Supervised_Homography_Learning_with_Realistic_Dataset_Generation_ICCV_2023_paper.pdf)] [[Code](https://github.com/JianghaiSCU/RealSH)]
  - keyword: realistic dataset generation
- ![paper](https://img.shields.io/badge/Dataset-red) ![page](https://img.shields.io/badge/Pretrain-model-blue) Semi-supervised Deep Large-Baseline Homography Estimation with Progressive Equivalence Constraint, `AAAI`. [[Paper](https://ojs.aaai.org/index.php/AAAI/article/download/25183/24955)] [[Code](https://github.com/megvii-research/LBHomo)]
  - keyword: large baseline dataset
- Exploring Progressive Hybrid-Degraded Image Processing for Homography Estimation, `ICASSP`. [[Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10096730&casa_token=fMSrZF-OOuIAAAAA:i24gOQjdCdsf22y-cQEwCmlNa84s0gXE2-lfWHEbxUnj1L6n_jKyu2EiHUr_rvqESRrH36Of&tag=1)]
- Recurrent homography estimation using homography-guided image warping and focus transformer,  `CVPR`. [[Paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Cao_Recurrent_Homography_Estimation_Using_Homography-Guided_Image_Warping_and_Focus_Transformer_CVPR_2023_paper.pdf)] [[Code](https://github.com/imdumpl78/RHWF)]
- Bilevel Progressive Homography Estimation Via Correlative Region-Focused Transformer, `SSRN`. [[Paper](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4625861)]
- Coarse-to-Fine Homography Estimation for Infrared and Visible Images, `Electronics`. [[Paper](https://www.mdpi.com/2079-9292/12/21/4441)]
- Mask-Homo: Pseudo Plane Mask-Guided Unsupervised Multi-Homography Estimation, `AAAI`. [[Paper](https://ojs.aaai.org/index.php/AAAI/article/view/28379)] [[Code](https://github.com/SAITPublic/MaskHomo)]
- Unsupervised deep homography with multi‐scale global attention, `IET Image Processing`. [[Paper](https://ietresearch.onlinelibrary.wiley.com/doi/full/10.1049/ipr2.12842)]
- Homography Estimation in Complex Topological Scenes, `arXiv`. [[Paper](https://arxiv.org/pdf/2308.01086)]
- Infrared and Visible Image Homography Estimation Based on Feature Correlation Transformers for Enhanced 6G Space–Air–Ground Integrated Network Perception, `Remote Sensing`. [[Paper](https://www.mdpi.com/2072-4292/15/14/3535)]
- Geometrized transformer for self-supervised homography estimation, `ICCV`. [[Paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Liu_Geometrized_Transformer_for_Self-Supervised_Homography_Estimation_ICCV_2023_paper.pdf)] [[Code](https://github.com/ruc-aimc-lab/GeoFormer)]

#### 2022

- Iterative deep homography estimation, `CVPR`. [[Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Cao_Iterative_Deep_Homography_Estimation_CVPR_2022_paper.pdf)] [[Code](https://github.com/imdumpl78/IHN)]
- ![page](https://img.shields.io/badge/Pretrain-model-blue) Unsupervised Homography Estimation with Coplanarity-Aware GAN, `CVPR`. [[Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Hong_Unsupervised_Homography_Estimation_With_Coplanarity-Aware_GAN_CVPR_2022_paper.pdf)] [[Code](https://github.com/megvii-research/HomoGAN)]
  - keyword: unsupervised homography estimation
- Content-Aware Unsupervised Deep Homography Estimation and its Extensions, `TPAMI`. [[Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9771389&casa_token=djckIfXYqkwAAAAA:JMgJoTIIS3fdc6yhjrOIhyepJpqHNZSfpa81XQbCeW4oMvV9Mm38ayLqklpdrQcWvcL2Dy7O)]
  - Keyword: journal version
- Learning to Generate High-Quality Images for Homography Estimation, `ICIP`. [[Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9897392&casa_token=LceLhra9-rwAAAAA:U7AQQhZcjLNM4ZMqsnAGfxv_iXNA6v9Ghof4dfBWNMTrWFQV9zYQxK3OmmdFj4kadzAE1Rxl)]
- Detail-Aware Deep Homography Estimation for Infrared and Visible Image, `Electronics`. [[Paper](https://www.mdpi.com/2079-9292/11/24/4185)]
- Towards a unified approach to homography estimation using image features and pixel intensities, `arXiv`. [[Paper](https://arxiv.org/pdf/2202.09716)]

#### 2021

- Motion Basis Learning for Unsupervised Deep Homography Estimation with Subspace Projection, `ICCV`. [[Paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Ye_Motion_Basis_Learning_for_Unsupervised_Deep_Homography_Estimation_With_Subspace_ICCV_2021_paper.pdf)] [[Code](https://github.com/megvii-research/BasesHomo)]
  - keyword: unsupervised homography estimation
- ![paper](https://img.shields.io/badge/Dataset-red) ![page](https://img.shields.io/badge/Pretrain-model-blue) Deep Lucas-Kanade Homography for Multimodal Image Alignment, `CVPR`. [[Paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Zhao_Deep_Lucas-Kanade_Homography_for_Multimodal_Image_Alignment_CVPR_2021_paper.pdf)] [[Code](https://github.com/placeforyiming/CVPR21-Deep-Lucas-Kanade-Homography)]
  - keyword: supervised homography estimation
- ![page](https://img.shields.io/badge/Pretrain-model-blue) Depth-aware multi-grid deep homography estimation with contextual correlation, `TCSVT`. [[Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9605632&casa_token=lop5MaMA40cAAAAA:ypyRTM-TzDD7-tVTJ7ndk0fT6zxNdmVvjtz4DD6uSrE-qjjagoEKdsy6NcRP49GGItZC_li5)] [[Code](https://github.com/nie-lang/Multi-Grid-Deep-Homography)]
- Deep Homography Estimation based on Attention Mechanism, `ICSAI`. [[Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9664027)]
- Perceptual loss for robust unsupervised homography estimation, `CVPRw`. [[Paper](https://openaccess.thecvf.com/content/CVPR2021W/IMW/papers/Koguciuk_Perceptual_Loss_for_Robust_Unsupervised_Homography_Estimation_CVPRW_2021_paper.pdf)]

#### 2020

- ![paper](https://img.shields.io/badge/Dataset-red) Deep homography estimation for dynamic scenes, `CVPR`. [[Paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Le_Deep_Homography_Estimation_for_Dynamic_Scenes_CVPR_2020_paper.pdf)] [[Code](https://github.com/lcmhoang/hmg-dynamics)]
- ![paper](https://img.shields.io/badge/Dataset-red) ![page](https://img.shields.io/badge/Pretrain-model-blue) Content-aware unsupervised deep homography estimation, `ECCV`. [[Paper](https://arxiv.org/pdf/1909.05983)] [[Code](https://github.com/JirongZhang/DeepHomography)]
  - keyword: small baseline dataset(CA-Unsupervised dataset)
- Homography Estimation with Convolutional Neural Networks Under Conditions of Variance, `arXiv`. [[Paper](https://arxiv.org/pdf/2010.01041)]
- Robust Homography Estimation via Dual Principal Component Pursuit, `CVPR`. [[Paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Ding_Robust_Homography_Estimation_via_Dual_Principal_Component_Pursuit_CVPR_2020_paper.pdf)]
- Self-supervised deep homography estimation with invertibility constraints, `PRL`. [[Paper](https://www.sciencedirect.com/science/article/pii/S0167865519302673)]
- SRHEN: Stepwise-Refining Homography Estimation Network via Parsing Geometric Correspondences in Deep Latent Space, `MM`. [[Paper](https://dl.acm.org/doi/pdf/10.1145/3394171.3413870)]
- CorNet: Unsupervised Deep Homography Estimation for Agricultural Aerial Imagery, `ECCV`. [[Paper](https://drive.google.com/file/d/1I6tpiodsdsnmt1g9P_cFdxFp7HN-O7UT/view)]

#### 2019

- STN-Homography: Direct estimation of homography parameters for image pairs, `Applied Sciences`. [[Paper](https://www.mdpi.com/2076-3417/9/23/5187)]
- Homography Estimation Based on Error Elliptical Distribution, `ICASSP`. [[Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8682180)]

#### 2018

- Unsupervised deep homography: A fast and robust homography estimation model, `RAL`. [[Paper](https://ieeexplore.ieee.org/document/8302515)]
  - Keyword: unsupervised homography estimation
- Rethinking Planar Homography Estimation Using Perspective Fields, `ACCV`. [[Paper](https://eprints.qut.edu.au/126933/1/0654.pdf)] 

#### 2017

- Homography estimation from image pairs with hierarchical convolutional networks, `ICCVw`. [[Paper](https://openaccess.thecvf.com/content_ICCV_2017_workshops/papers/w17/Nowruzi_Homography_Estimation_From_ICCV_2017_paper.pdf)]

#### 2016

- Deep image homography estimation, `arXiv`. [[Paper](https://arxiv.org/pdf/1606.03798)] [[Unofficial_Code1](https://github.com/yishiliuhuasheng/deep_image_homography_estimation)] [[Unofficial_Code2](https://github.com/paeccher/Deep-Homography-Estimation-Pytorch)]
  - Keyword: first learning-based supervised homography estimation method, realistic dataset generation

#### 2014

- HEASK: Robust homography estimation based on appearance similarity and keypoint correspondences, `PR`. [[Paper](https://www.sciencedirect.com/science/article/pii/S0031320313002112)]

#### 2009

- Homography estimation, [[Paper](https://www.cs.ubc.ca/sites/default/files/2022-12/Dubrofsky_Elan.pdf)]

## Image Alignment

| Date |                             Ref                              | Paper Title                                                  |    Type    | Code                                                         |
| :--: | :----------------------------------------------------------: | :----------------------------------------------------------- | :--------: | ------------------------------------------------------------ |
| 2025 | Journal of King Saud University Computer and Information Sciences | [Hierarchical grid-constrained fusion network for image stitching](https://link.springer.com/article/10.1007/s44443-025-00005-6) | Grid-based | https://github.com/albestobe/HGFN                            |
| 2024 |                           `Access`                           | [MGHE-Net: A Transformer-Based Multi-Grid Homography Estimation Network for Image Stitching](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10489948) | Grid-based |                                                              |
| 2024 |                           `arXiv`                            | [Parallax-tolerant Image Stitching via Segmentation-guided Multi-homography Warping](https://arxiv.org/pdf/2406.19922) |            |                                                              |
| 2023 |                            `CVPR`                            | [PRISE: Demystifying Deep Lucas-Kanade with Strongly Star-Convex Constraints for Multimodel Image Alignment](https://openaccess.thecvf.com/content/CVPR2023/papers/Zhang_PRISE_Demystifying_Deep_Lucas-Kanade_With_Strongly_Star-Convex_Constraints_for_Multimodel_CVPR_2023_paper.pdf) | Flow-based | [Code](https://github.com/swiftzhang125/PRISE)               |
| 2023 |                            `ICCV`                            | [Parallax-Tolerant Unsupervised Deep Image Stitching](https://openaccess.thecvf.com/content/ICCV2023/papers/Nie_Parallax-Tolerant_Unsupervised_Deep_Image_Stitching_ICCV_2023_paper.pdf) |            | [Code](https://github.com/nie-lang/UDIS2)                    |
| 2022 |                           `arXiv`                            | [Warped Convolutional Networks: Bridge Homography to sl(3) algebra by Group Convolution](https://arxiv.org/pdf/2206.11657) |            |                                                              |
| 2021 |                           `CICAI`                            | [Unsupervised Deep Plane-Aware Multi-homography Learning for Image Alignment](https://link.springer.com/chapter/10.1007/978-3-030-93046-2_45) |            |                                                              |
| 2021 |                            `CVPR`                            | [Localtrans: A multiscale local transformer network for cross-resolution homography estimation](https://openaccess.thecvf.com/content/ICCV2021/papers/Shao_LocalTrans_A_Multiscale_Local_Transformer_Network_for_Cross-Resolution_Homography_Estimation_ICCV_2021_paper.pdf) |            |                                                              |
| 2021 |                            `TIP`                             | [Unsupervised deep image stitching: Reconstructing stitched features to images](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9472883) |            |                                                              |
| 2020 |                            `TIP`                             | [Cross-Weather Image Alignment via Latent Generative Model With Intensity Consistency](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9046236&casa_token=hVGip0b16QAAAAAA:6DucWFNMk96qPsGxf_B9l1dKctCyNB4LNtb1fjsdhInXR2w3MGS9WflinZCK8TlLTvQ2-Y0y) |            |                                                              |
| 2020 |  `Journal of Visual Communication and Image Representation`  | [A view-free image stitching network based on global homography](https://www.sciencedirect.com/science/article/pii/S1047320320301784?casa_token=j4oKVYUdERcAAAAA:NvpUUuh4sK_sfz2eaD8IcfwPcIzIMTkwAo0wDC6A90713r_DxxUnvKZfwhZx2C4U5nmQuR7XUg) |            | [Code](https://github.com/nie-lang/DeepImageStitching-1.0)   |
| 2020 |                           `arXiv`                            | [Learning edge-preserved image stitching from large-baseline deep homography](https://arxiv.org/pdf/2012.06194) |            |                                                              |
| 2020 |                            `CVPR`                            | [Warping Residual Based Image Stitching for Large Parallax](https://openaccess.thecvf.com/content_CVPR_2020/papers/Lee_Warping_Residual_Based_Image_Stitching_for_Large_Parallax_CVPR_2020_paper.pdf) |            |                                                              |
| 2020 |                            `ECCV`                            | [Ransac-flow: generic two-stage image alignment](https://par.nsf.gov/servlets/purl/10202962) | Flow-based | [Code](https://github.com/XiSHEN0220/RANSAC-Flow) [Project](https://imagine.enpc.fr/~shenx/RANSAC-Flow/) |
| 2019 |                           `arXiv`                            | [DeepMeshFlow: Content adaptive mesh deformation for robust image registration](https://arxiv.org/pdf/1912.05131) | Grid-based |                                                              |
| 2018 |                            `ECCV`                            | [Multimodal image alignment through a multiscale chain of neural networks with application to re mote sensing](https://openaccess.thecvf.com/content_ECCV_2018/papers/Armand_Zampieri_Multimodal_image_alignment_ECCV_2018_paper.pdf) |            |                                                              |
| 2017 |                            `CVPR`                            | [CLKN: Cascaded Lucas–Kanade networks for image alignment](https://openaccess.thecvf.com/content_cvpr_2017/papers/Chang_CLKN_Cascaded_Lucas-Kanade_CVPR_2017_paper.pdf) |            |                                                              |
| 2015 |                            `IJCV`                            | [Rationalizing Efficient Compositional Image Alignment](https://link.springer.com/article/10.1007/s11263-014-0769-6) |            |                                                              |
| 2014 |                            `CVPR`                            | [Parallax-tolerant Image Stitching](https://www.cv-foundation.org/openaccess/content_cvpr_2014/papers/Zhang_Parallax-tolerant_Image_Stitching_2014_CVPR_paper.pdf) |            |                                                              |
| 2007 |  `Foundations and Trends® in Computer Graphics and Vision`   | [Image alignment and stitching: A tutorial](https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=340fe67d687ddbaae5a7b9a0653e6ff99f340830) |  Tutorial  |                                                              |                      |

## Application

| Date |               Ref                | Paper Title                                                  |                             Code                             |
| :--: | :------------------------------: | :----------------------------------------------------------- | :----------------------------------------------------------: |
| 2024 |              `AAAI`              | [Deep Homography Estimation for Visual Place Recognition](https://arxiv.org/pdf/2402.16086v1) |          [Code](https://github.com/Lu-Feng/DHE-VPR)          |
| 2024 |             `arXiv`              | [STHN: Deep Homography Estimation for UAV Thermal Geo-localization with Satellite Imagery](https://arxiv.org/pdf/2405.20470) | [Code](https://github.com/arplaboratory/STHN) [Project](https://xjh19971.github.io/STHN/) |
| 2023 |               `IV`               | [Homography Estimation for Camera Calibration in Complex Topological Scenes](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10186786) |                                                              |
| 2022 |              arXiv               | [SSORN: Self-Supervised Outlier Removal Network for Robust Homography Estimation](https://arxiv.org/pdf/2208.14093) |                                                              |
| 2022 | `Symposium on Applied Computing` | [Semi-Supervised Learning for Image Alignment in Teach and Repeat navigation](https://dl.acm.org/doi/pdf/10.1145/3477314.3507045) |                                                              |
| 2021 |             `arXiv`              | [Weather GAN: Multi-Domain Weather Translation Using Generative Adversarial Networks](https://arxiv.org/pdf/2103.05422) |                                                              |
| 2021 |              `CVPR`              | [Deep homography for efficient stereo image compression](https://openaccess.thecvf.com/content/CVPR2021/papers/Deng_Deep_Homography_for_Efficient_Stereo_Image_Compression_CVPR_2021_paper.pdf) |        [Code](https://github.com/ywz978020607/HESIC)         |





