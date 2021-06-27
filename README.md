# SkeletonNet

This repository constains the codes and [ShapeNetV1-Surface-Skeleton](https://drive.google.com/file/d/1FlXiWFuBbryyNvyH07kGGl9WlmuYPVAP/view?usp=sharing),[ShapNetV1-SkeletalVolume](https://drive.google.com/file/d/1gmT6wF-wLYoa_CWfNsPYd0QtwW0V9NqB/view?usp=sharing) and 2d image datasets [ShapeNetRendering](http://cvgl.stanford.edu/data2/ShapeNetRendering.tgz).
Please download the above datasets at the first,  and then put them under the ```SkeletonNet/sharedata``` folder.

## Prepare Skeleton points/volumes

* If you want to use our skeletal point cloud extraction code, you can download the [skeleton extraction code](https://drive.google.com/file/d/1SGL8LJl1kgtUzM8_COwMMo-SzCPSpNLz/view?usp=sharing). This code is built on Visual Studio2013 + Qt. 
* If you want to convert the skeletal point clouds to skeletal volumes, you can run the below scripts.

```shell 
python sharedata/prepare_skeletalvolume.py --cats 03001627 --vx_res 32
python sharedata/prepare_skeletalvolume2.py --cats 03001627 --vx_res 64
python sharedata/prepare_skeletalvolume2.py --cats 03001627 --vx_res 128
python sharedata/prepare_skeletalvolume2.py --cats 03001627 --vx_res 256
```

Before running above scripts, you need to change ```raw_pointcloud_dir and upsample_skeleton_dir``` used when extracting skeletal points.

## Installation

First you need to create an anaconda environment called SkeletonNet using
```shell
conda env create -f environment.yaml
conda activate SkeletonNet
```

## Implementation details

For each stage, please follow the README.md under the ```Skeleton_Inference/SkeGCNN/SkeDISN``` folder.


## Demo



## Evalation 

Please refer to the README.md under the ```./SkeDISN``` folder.

## Citation
If you find this work useful in your research, please consider citing:
```shell
@InProceedings{Tang_2019_CVPR,
author = {Tang, Jiapeng and Han, Xiaoguang and Pan, Junyi and Jia, Kui and Tong, Xin},
title = {A Skeleton-Bridged Deep Learning Approach for Generating Meshes of Complex Topologies From Single RGB Images},
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2019}
}

@article{tang2020skeletonnet,
  title={SkeletonNet: A Topology-Preserving Solution for Learning Mesh Reconstruction of Object Surfaces from RGB Images},
  author={Tang, Jiapeng and Han, Xiaoguang and Tan, Mingkui and Tong, Xin and Jia, Kui},
  journal={arXiv preprint arXiv:2008.05742},
  year={2020}
}
```

## Contact 
If you have any questions,  please feel free to contact with Tang Jiapeng msjptang@mail.scut.edu.cn.