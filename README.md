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

#### 2024

- **DMHomo**: Learning Homography with Diffusion Model, *ToG*. [[Paper](https://dl.acm.org/doi/full/10.1145/3652207)] [[Code](https://github.com/lhaippp/DMHomo)]

- **CrossHomo**: Cross-Modality and Cross-Resolution Homography Estimation, *TPAMI*. [[Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10438073&casa_token=92CU3CFbsXIAAAAA:DJN1QCKXfj7pFYlSD4sLh33HUtBQ-BjhiBBRtBtUbcEHjKnSyduvywgyYvAnI9UJW7MkOboG&tag=1)] [[Code](https://github.com/lep990816/CrossHomo)]

#### 2023

- Edge-aware Correlation Learning for Unsupervised Progressive Homography Estimation, *TCSVT*. [[Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10314523&casa_token=86gKd0-dQBUAAAAA:VJbu9XBitO-BQPYhu2uwW0FI5EMaH2YshfkBnuoHC4jtkCWN8jFEFm2A2Y_Bd5-DU0SChWJP)]

- Unsupervised Global and Local Homography Estimation with Motion Basis Learning, *TPAMI*. [[Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9956874&casa_token=-Tp9JpW764gAAAAA:NkiBmmnJ3WHrxXjm5DeEFFXna8haxfiG52I5pWptORdbvSeU2vVZtDl8Y95jbkiIJYnSk2ql)] [[Code](https://github.com/megvii-research/BasesHomo)]

- Supervised Homography Learning with Realistic Dataset Generation, *ICCV*. [[Paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Jiang_Supervised_Homography_Learning_with_Realistic_Dataset_Generation_ICCV_2023_paper.pdf)] [[Code](https://github.com/JianghaiSCU/RealSH)]
  - Keyword: realistic dataset generation;

- Semi-supervised Deep Large-Baseline Homography Estimation with Progressive Equivalence Constraint, *AAAI*. [[Paper](https://ojs.aaai.org/index.php/AAAI/article/download/25183/24955)] [[Code](https://github.com/megvii-research/LBHomo)]
  - Keyword: large baseline dataset;
- Supervised Homography Learning with Realistic Dataset Generation, *ICCV*. [[Paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Jiang_Supervised_Homography_Learning_with_Realistic_Dataset_Generation_ICCV_2023_paper.pdf)] [[Code](https://github.com/JianghaiSCU/RealSH)]

#### 2022

- Iterative deep homography estimation, *CVPR*. [[Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Cao_Iterative_Deep_Homography_Estimation_CVPR_2022_paper.pdf)] [[Code](https://github.com/imdumpl78/IHN)]

- Unsupervised Homography Estimation with Coplanarity-Aware GAN, *CVPR*. [[Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Hong_Unsupervised_Homography_Estimation_With_Coplanarity-Aware_GAN_CVPR_2022_paper.pdf)] [[Code](https://github.com/megvii-research/HomoGAN)]

- Content-Aware Unsupervised Deep Homography Estimation and its Extensions, *TPAMI*. [[Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9771389&casa_token=djckIfXYqkwAAAAA:JMgJoTIIS3fdc6yhjrOIhyepJpqHNZSfpa81XQbCeW4oMvV9Mm38ayLqklpdrQcWvcL2Dy7O)]
  - Keyword: journal version;

#### 2021

- Motion Basis Learning for Unsupervised Deep Homography Estimation with Subspace Projection, *ICCV*. [[Paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Ye_Motion_Basis_Learning_for_Unsupervised_Deep_Homography_Estimation_With_Subspace_ICCV_2021_paper.pdf)] [[Code](https://github.com/megvii-research/BasesHomo)]
- Deep Lucas-Kanade Homography for Multimodal Image Alignment, *CVPR*. [[Paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Zhao_Deep_Lucas-Kanade_Homography_for_Multimodal_Image_Alignment_CVPR_2021_paper.pdf)] [[Code](https://github.com/placeforyiming/CVPR21-Deep-Lucas-Kanade-Homography)]
- Depth-aware multi-grid deep homography estimation with contextual correlation. [[Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9605632&casa_token=lop5MaMA40cAAAAA:ypyRTM-TzDD7-tVTJ7ndk0fT6zxNdmVvjtz4DD6uSrE-qjjagoEKdsy6NcRP49GGItZC_li5)] [[Code](https://github.com/nie-lang/Multi-Grid-Deep-Homography)]

#### 2020

- Deep homography estimation for dynamic scenes. [[Paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Le_Deep_Homography_Estimation_for_Dynamic_Scenes_CVPR_2020_paper.pdf)] [[Code](https://github.com/lcmhoang/hmg-dynamics)]
- Content-aware unsupervised deep homography estimation, *ECCV*. [[Paper](https://arxiv.org/pdf/1909.05983)] [[Code](https://github.com/JirongZhang/DeepHomography)]
  - keyword: small baseline dataset;

#### 2019

#### 2018

- **Unsupervised deep homography**: A fast and robust homography estimation model, *RAL*. [[Paper](https://ieeexplore.ieee.org/document/8302515)] [[Unofficial_Code1]()]
  - Keyword:  

#### 2016

- Deep image homography estimation, *arXiv*. [[Paper](https://arxiv.org/pdf/1606.03798)] [[Unofficial_Code1](https://github.com/yishiliuhuasheng/deep_image_homography_estimation)] [[Unofficial_Code2](https://github.com/paeccher/Deep-Homography-Estimation-Pytorch)]
  - Keyword: frist learning-based homography estimation method; realistic dataset generation; 

## Image Alignment

#### 2023

- **PRISE**: Demystifying Deep Lucas-Kanade with Strongly Star-Convex Constraints for Multimodel Image Alignment. *CVPR*. [[Paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Zhang_PRISE_Demystifying_Deep_Lucas-Kanade_With_Strongly_Star-Convex_Constraints_for_Multimodel_CVPR_2023_paper.pdf)] [[Code](https://github.com/swiftzhang125/PRISE)]

#### 2020

- Cross-Weather Image Alignment via Latent Generative Model With Intensity Consistency, *TIP*. [[Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9046236&casa_token=hVGip0b16QAAAAAA:6DucWFNMk96qPsGxf_B9l1dKctCyNB4LNtb1fjsdhInXR2w3MGS9WflinZCK8TlLTvQ2-Y0y)]
- A view-free image stitching network based on global homography, *Journal of Visual Communication and Image Representation*. [[Paper](https://www.sciencedirect.com/science/article/pii/S1047320320301784?casa_token=j4oKVYUdERcAAAAA:NvpUUuh4sK_sfz2eaD8IcfwPcIzIMTkwAo0wDC6A90713r_DxxUnvKZfwhZx2C4U5nmQuR7XUg)] [[Code](https://github.com/nie-lang/DeepImageStitching-1.0)]

## Application

#### 2024

- Deep Homography Estimation for Visual Place Recognition, *AAAI*. [[Paper](https://arxiv.org/pdf/2402.16086v1)] [[Code](https://github.com/Lu-Feng/DHE-VPR)]

#### 2021

- **Weather GAN**: Multi-Domain Weather Translation Using Generative Adversarial Networks, *arXiv*. [[Paper](https://arxiv.org/pdf/2103.05422)] [[Code]()]
