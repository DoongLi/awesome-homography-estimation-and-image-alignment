# awesome-homography-estimation-and-image-alignment[![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)

This repo contains a curative list of **homography estimation, image alignment and related application**.

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

#### 2024

- ![paper](https://img.shields.io/badge/Dataset-red)![page](https://img.shields.io/badge/Pretrain-model-blue) **DMHomo**: Learning Homography with Diffusion Model, `ToG`. [[Paper](https://dl.acm.org/doi/full/10.1145/3652207)] [[Code](https://github.com/lhaippp/DMHomo)]
  - keyword: realistic dataset generation, supervised homography estimation;
- ![page](https://img.shields.io/badge/Pretrain-model-blue) **CrossHomo**: Cross-Modality and Cross-Resolution Homography Estimation, `TPAMI`. [[Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10438073&casa_token=92CU3CFbsXIAAAAA:DJN1QCKXfj7pFYlSD4sLh33HUtBQ-BjhiBBRtBtUbcEHjKnSyduvywgyYvAnI9UJW7MkOboG&tag=1)] [[Code](https://github.com/lep990816/CrossHomo)]
  - keyword: cross-modality images, cross-resolution images;
- ![page](https://img.shields.io/badge/Pretrain-model-blue) **Gyroflow+**: Gyroscope-guided unsupervised deep homography and optical flow learning, `IJCV`. [[Paper](https://link.springer.com/article/10.1007/s11263-023-01978-5)] [[Code](https://github.com/lhaippp/GyroFlowPlus)]
- Unsupervised Homography Estimation with Pixel-level SVDD, `TCSVT`. [[Paper](https://ieeexplore.ieee.org/abstract/document/10510339)]
- **AbHE**: All Attention-Based Homography Estimation, `TIM`. [[Paper](https://ieeexplore.ieee.org/abstract/document/10472928)] [[Code](https://github.com/mingxiaohuo/ABHE)]
- ![paper](https://img.shields.io/badge/Dataset-red) Analyzing the Domain Shift Immunity of Deep Homography Estimation, `WACV`. [[Paper](https://openaccess.thecvf.com/content/WACV2024/papers/Shao_Analyzing_the_Domain_Shift_Immunity_of_Deep_Homography_Estimation_WACV_2024_paper.pdf)] [[Code](https://github.com/MingzhenShao/Homography_estimation)]

#### 2023

- Edge-aware Correlation Learning for Unsupervised Progressive Homography Estimation, `TCSVT`. [[Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10314523&casa_token=86gKd0-dQBUAAAAA:VJbu9XBitO-BQPYhu2uwW0FI5EMaH2YshfkBnuoHC4jtkCWN8jFEFm2A2Y_Bd5-DU0SChWJP)]
  - keyword: unsupervised homography estimation;
- ![page](https://img.shields.io/badge/Pretrain-model-blue) Unsupervised Global and Local Homography Estimation with Motion Basis Learning, `TPAMI`. [[Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9956874&casa_token=-Tp9JpW764gAAAAA:NkiBmmnJ3WHrxXjm5DeEFFXna8haxfiG52I5pWptORdbvSeU2vVZtDl8Y95jbkiIJYnSk2ql)] [[Code](https://github.com/megvii-research/BasesHomo)]
  - keyword: unsupervised homography estimation;
- ![paper](https://img.shields.io/badge/Dataset-red)![page](https://img.shields.io/badge/Pretrain-model-blue) Supervised Homography Learning with Realistic Dataset Generation, `ICCV`. [[Paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Jiang_Supervised_Homography_Learning_with_Realistic_Dataset_Generation_ICCV_2023_paper.pdf)] [[Code](https://github.com/JianghaiSCU/RealSH)]
  - keyword: realistic dataset generation;
- ![paper](https://img.shields.io/badge/Dataset-red)![page](https://img.shields.io/badge/Pretrain-model-blue) Semi-supervised Deep Large-Baseline Homography Estimation with Progressive Equivalence Constraint, `AAAI`. [[Paper](https://ojs.aaai.org/index.php/AAAI/article/download/25183/24955)] [[Code](https://github.com/megvii-research/LBHomo)]
  - keyword: large baseline dataset;
- ![page](https://img.shields.io/badge/Pretrain-model-blue) Supervised Homography Learning with Realistic Dataset Generation, `ICCV`. [[Paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Jiang_Supervised_Homography_Learning_with_Realistic_Dataset_Generation_ICCV_2023_paper.pdf)] [[Code](https://github.com/JianghaiSCU/RealSH)]
  - keyword: realistic dataset generation;
- Exploring Progressive Hybrid-Degraded Image Processing for Homography Estimation, `ICASSP`. [[Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10096730&casa_token=fMSrZF-OOuIAAAAA:i24gOQjdCdsf22y-cQEwCmlNa84s0gXE2-lfWHEbxUnj1L6n_jKyu2EiHUr_rvqESRrH36Of&tag=1)]
- Recurrent homography estimation using homography-guided image warping and focus transformer,  `CVPR`. [[Paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Cao_Recurrent_Homography_Estimation_Using_Homography-Guided_Image_Warping_and_Focus_Transformer_CVPR_2023_paper.pdf)] [[Code](https://github.com/imdumpl78/RHWF)]
- Bilevel Progressive Homography Estimation Via Correlative Region-Focused Transformer, `SSRN`. [[Paper](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4625861)]
- Coarse-to-Fine Homography Estimation for Infrared and Visible Images, `Electronics`. [[Paper](https://www.mdpi.com/2079-9292/12/21/4441)]
- Mask-Homo: Pseudo Plane Mask-Guided Unsupervised Multi-Homography Estimation, `AAAI`. [[Paper](https://ojs.aaai.org/index.php/AAAI/article/view/28379)] [[Code](https://github.com/SAITPublic/MaskHomo)]
- Unsupervised deep homography with multi‐scale global attention, `IET Image Processing`. [[Paper](https://ietresearch.onlinelibrary.wiley.com/doi/full/10.1049/ipr2.12842)]
- Homography Estimation in Complex Topological Scenes, `arXiv`. [[Paper](https://arxiv.org/pdf/2308.01086)]

#### 2022

- Iterative deep homography estimation, `CVPR`. [[Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Cao_Iterative_Deep_Homography_Estimation_CVPR_2022_paper.pdf)] [[Code](https://github.com/imdumpl78/IHN)]
- ![page](https://img.shields.io/badge/Pretrain-model-blue) Unsupervised Homography Estimation with Coplanarity-Aware GAN, `CVPR`. [[Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Hong_Unsupervised_Homography_Estimation_With_Coplanarity-Aware_GAN_CVPR_2022_paper.pdf)] [[Code](https://github.com/megvii-research/HomoGAN)]
  - keyword: unsupervised homography estimation;
- Content-Aware Unsupervised Deep Homography Estimation and its Extensions, `TPAMI`. [[Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9771389&casa_token=djckIfXYqkwAAAAA:JMgJoTIIS3fdc6yhjrOIhyepJpqHNZSfpa81XQbCeW4oMvV9Mm38ayLqklpdrQcWvcL2Dy7O)]
  - Keyword: journal version;
- Learning to Generate High-Quality Images for Homography Estimation, `ICIP`. [[Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9897392&casa_token=LceLhra9-rwAAAAA:U7AQQhZcjLNM4ZMqsnAGfxv_iXNA6v9Ghof4dfBWNMTrWFQV9zYQxK3OmmdFj4kadzAE1Rxl)]
- Detail-Aware Deep Homography Estimation for Infrared and Visible Image, `Electronics`. [[Paper](https://www.mdpi.com/2079-9292/11/24/4185)]
- Towards a unified approach to homography estimation using image features and pixel intensities, `arXiv`. [[Paper](https://arxiv.org/pdf/2202.09716)]

#### 2021

- Motion Basis Learning for Unsupervised Deep Homography Estimation with Subspace Projection, `ICCV`. [[Paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Ye_Motion_Basis_Learning_for_Unsupervised_Deep_Homography_Estimation_With_Subspace_ICCV_2021_paper.pdf)] [[Code](https://github.com/megvii-research/BasesHomo)]
  - keyword: unsupervised homography estimation;
- ![paper](https://img.shields.io/badge/Dataset-red)![page](https://img.shields.io/badge/Pretrain-model-blue) Deep Lucas-Kanade Homography for Multimodal Image Alignment, `CVPR`. [[Paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Zhao_Deep_Lucas-Kanade_Homography_for_Multimodal_Image_Alignment_CVPR_2021_paper.pdf)] [[Code](https://github.com/placeforyiming/CVPR21-Deep-Lucas-Kanade-Homography)]
  - keyword: supervised homography estimation;
- ![page](https://img.shields.io/badge/Pretrain-model-blue) Depth-aware multi-grid deep homography estimation with contextual correlation, `TCSVT`. [[Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9605632&casa_token=lop5MaMA40cAAAAA:ypyRTM-TzDD7-tVTJ7ndk0fT6zxNdmVvjtz4DD6uSrE-qjjagoEKdsy6NcRP49GGItZC_li5)] [[Code](https://github.com/nie-lang/Multi-Grid-Deep-Homography)]
- Deep Homography Estimation based on Attention Mechanism, `ICSAI`. [[Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9664027)]

#### 2020

- ![paper](https://img.shields.io/badge/Dataset-red) Deep homography estimation for dynamic scenes, `CVPR`. [[Paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Le_Deep_Homography_Estimation_for_Dynamic_Scenes_CVPR_2020_paper.pdf)] [[Code](https://github.com/lcmhoang/hmg-dynamics)]
- ![paper](https://img.shields.io/badge/Dataset-red) ![page](https://img.shields.io/badge/Pretrain-model-blue) Content-aware unsupervised deep homography estimation, `ECCV`. [[Paper](https://arxiv.org/pdf/1909.05983)] [[Code](https://github.com/JirongZhang/DeepHomography)]
  - keyword: small baseline dataset(CA-Unsupervised dataset);
- Homography Estimation with Convolutional Neural Networks Under Conditions of Variance, `arXiv`. [[Paper](https://arxiv.org/pdf/2010.01041)]
- Robust Homography Estimation via Dual Principal Component Pursuit, `CVPR`. [[Paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Ding_Robust_Homography_Estimation_via_Dual_Principal_Component_Pursuit_CVPR_2020_paper.pdf)]
- Self-supervised deep homography estimation with invertibility constraints, `PRL`. [[Paper](https://www.sciencedirect.com/science/article/pii/S0167865519302673)]
- **SRHEN**: Stepwise-Refining Homography Estimation Network via Parsing Geometric Correspondences in Deep Latent Space, `MM`. [[Paper](https://dl.acm.org/doi/pdf/10.1145/3394171.3413870)]
- **CorNet**: Unsuper vised Deep Homography Estimation for Agricultural Aerial Imagery, `ECCV`. [[Paper](https://drive.google.com/file/d/1I6tpiodsdsnmt1g9P_cFdxFp7HN-O7UT/view)]

#### 2019

- **STN-Homography**: Direct estimation of homography parameters for image pairs, `Applied Sciences`. [[Paper](https://www.mdpi.com/2076-3417/9/23/5187)]
- Homography Estimation Based on Error Elliptical Distribution, `ICASSP`. [[Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8682180)]

#### 2018

- **Unsupervised deep homography**: A fast and robust homography estimation model, `RAL`. [[Paper](https://ieeexplore.ieee.org/document/8302515)] [[Unofficial_Code1]()]
  - Keyword: unsupervised homography estimation;
- Rethinking Planar Homography Estimation Using Perspective Fields, `ACCV`. [[Paper](https://eprints.qut.edu.au/126933/1/0654.pdf)] 

#### 2017

- Homography estimation from image pairs with hierarchical convolutional networks, `ICCV`. [[Paper](https://openaccess.thecvf.com/content_ICCV_2017_workshops/papers/w17/Nowruzi_Homography_Estimation_From_ICCV_2017_paper.pdf)]

#### 2016

- Deep image homography estimation, `arXiv`. [[Paper](https://arxiv.org/pdf/1606.03798)] [[Unofficial_Code1](https://github.com/yishiliuhuasheng/deep_image_homography_estimation)] [[Unofficial_Code2](https://github.com/paeccher/Deep-Homography-Estimation-Pytorch)]
  - Keyword: frist learning-based supervised homography estimation method; realistic dataset generation; 

#### 2014

- HEASK: Robust homography estimation based on appearance similarity and keypoint correspondences, `PR`. [[Paper](https://www.sciencedirect.com/science/article/pii/S0031320313002112)]

#### 2009

- Homography estimation, [[Paper](https://www.cs.ubc.ca/sites/default/files/2022-12/Dubrofsky_Elan.pdf)]

## Image Alignment

#### 2024

- **MGHE-Net**: A Transformer-Based Multi-Grid Homography Estimation Network for Image Stitching, `Access`. [[Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10489948)]

#### 2023

- **PRISE**: Demystifying Deep Lucas-Kanade with Strongly Star-Convex Constraints for Multimodel Image Alignment, `CVPR`. [[Paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Zhang_PRISE_Demystifying_Deep_Lucas-Kanade_With_Strongly_Star-Convex_Constraints_for_Multimodel_CVPR_2023_paper.pdf)] [[Code](https://github.com/swiftzhang125/PRISE)]
- Parallax-Tolerant Unsupervised Deep Image Stitching, `ICCV`. [[Paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Nie_Parallax-Tolerant_Unsupervised_Deep_Image_Stitching_ICCV_2023_paper.pdf)] [[Code](https://github.com/nie-lang/UDIS2)]

#### 2022

- Warped Convolutional Networks: Bridge Homography to sl(3) algebra by Group Convolution, `arXiv`. [[Paper](https://arxiv.org/pdf/2206.11657)]

#### 2021

- Unsupervised Deep Plane-Aware Multi-homography Learning for Image Alignment, `CICAI`. [[Paper](https://link.springer.com/chapter/10.1007/978-3-030-93046-2_45)]
- **Localtrans**: A multiscale local transformer network for cross-resolution homography estimation, `CVPR`. [[Paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Shao_LocalTrans_A_Multiscale_Local_Transformer_Network_for_Cross-Resolution_Homography_Estimation_ICCV_2021_paper.pdf)]

#### 2020

- Cross-Weather Image Alignment via Latent Generative Model With Intensity Consistency, `TIP`. [[Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9046236&casa_token=hVGip0b16QAAAAAA:6DucWFNMk96qPsGxf_B9l1dKctCyNB4LNtb1fjsdhInXR2w3MGS9WflinZCK8TlLTvQ2-Y0y)]
- A view-free image stitching network based on global homography, `Journal of Visual Communication and Image Representation`. [[Paper](https://www.sciencedirect.com/science/article/pii/S1047320320301784?casa_token=j4oKVYUdERcAAAAA:NvpUUuh4sK_sfz2eaD8IcfwPcIzIMTkwAo0wDC6A90713r_DxxUnvKZfwhZx2C4U5nmQuR7XUg)] [[Code](https://github.com/nie-lang/DeepImageStitching-1.0)]
- Learning edge-preserved image stitching from large-baseline deep homography, `arXiv`. [[Paper](https://arxiv.org/pdf/2012.06194)]
- Warping Residual Based Image Stitching for Large Parallax, `CVPR`. [[Paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Lee_Warping_Residual_Based_Image_Stitching_for_Large_Parallax_CVPR_2020_paper.pdf)]

#### 2019

- **DeepMeshFlow**: Content adaptive mesh deformation for robust image registration, `arXiv`. [[Paper](https://arxiv.org/pdf/1912.05131)]

#### 2018

- Multimodal image alignment through a multiscale chain of neural networks with application to re mote sensing, `ECCV`. [[Paper](https://openaccess.thecvf.com/content_ECCV_2018/papers/Armand_Zampieri_Multimodal_image_alignment_ECCV_2018_paper.pdf)]

#### 2017

- **CLKN**: Cascaded Lucas–Kanade networks for image alignment, `CVPR`. [[Paper](https://openaccess.thecvf.com/content_cvpr_2017/papers/Chang_CLKN_Cascaded_Lucas-Kanade_CVPR_2017_paper.pdf)]

#### 2015

- Rationalizing Efficient Compositional Image Alignment, `IJCV`. [[Paper](https://link.springer.com/article/10.1007/s11263-014-0769-6)]

## Application

#### 2024

- Deep Homography Estimation for Visual Place Recognition, `AAAI`. [[Paper](https://arxiv.org/pdf/2402.16086v1)] [[Code](https://github.com/Lu-Feng/DHE-VPR)]

#### 2023

- Homography Estimation for Camera Calibration in Complex Topological Scenes, `IV`. [[Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10186786)]

#### 2022

- **SSORN**: Self-Supervised Outlier Removal Network for Robust Homography Estimation, `arXiv`. [[Paper](https://arxiv.org/pdf/2208.14093)]

#### 2021

- **Weather GAN**: Multi-Domain Weather Translation Using Generative Adversarial Networks, `arXiv`. [[Paper](https://arxiv.org/pdf/2103.05422)] [[Code]()]
- Deep homography for efficient stereo image compression, `CVPR`. [[Paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Deng_Deep_Homography_for_Efficient_Stereo_Image_Compression_CVPR_2021_paper.pdf)] [[Code](https://github.com/ywz978020607/HESIC)]

