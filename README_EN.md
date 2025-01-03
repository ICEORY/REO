## Robust and Efficient 3D Semantic Occupancy Prediction with Calibration-free Spatial Transformation

[[中文](./README.md)|EN]

## Introduction

Existing methods mainly rely on spatial transformation with sensor calibration information. However, this operation also results in two significant issues: First, the model performance is degraded due to the calibration noise (e.g., changing of sensor position during driving ). Second, the spatial transformation is computationally expensive and limits the efficiency of existing methods. In this work, we exploit a calibration-free spatial transformation scheme based on the vanilla attention scheme, to project the 2D images and point clouds into a compact BEV plane. Besides, we also introduce extra auxiliary training tasks and an efficient decoder to improve the model performance and efficiency, respectively.

![image-20241118152142577](./asset/image-20241118152142577.png)

![image-20241118152142577](./asset/reo-smk-results.gif)

## Main Results

- Efficiency analysis on OpenOccupancy nuScenes

  ![image-20241118152418778](./asset/image-20241118152418778.png)

- Results on OpenOccupancy nuScenes 

![image-20241118152243189](./asset/image-20241118152243189.png)

- Results on OCC3D-nuScenes

  ![image-20241118152315972](./asset/image-20241118152315972.png)

## Results on Test Sets

- [SemanticKITTI Scene Completion](https://codalab.lisn.upsaclay.fr/my/competition/submission/869748/detailed_results/)

  ![image-20250103163319139](./asset/image-20250103163319139.png)

- [OCC3D-nuScenes](https://eval.ai/web/challenges/challenge-page/2045/leaderboard/4838) 

![img_v3_02i6_aa614cbb-0128-43ea-9822-6af651c45adg](./asset/img_v3_02i6_aa614cbb-0128-43ea-9822-6af651c45adg.jpg)


## How to use

This work is based on our prior work, i.e., [PMF](https://github.com/ICEORY/PMF). If you are familiar with our PMF project, you can run this project easily.

The code of this project will be released after our work is finally accepted.

## Citation

```
@article{zhuang2024robust3dsemanticoccupancy,
      title={Robust 3D Semantic Occupancy Prediction with Calibration-free Spatial Transformation}, 
      author={Zhuangwei Zhuang and Ziyin Wang and Sitao Chen and Lizhao Liu and Hui Luo and Mingkui Tan},
      journal={arXiv preprint arXiv:2411.12177},
      year={2024}
}
```

