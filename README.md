# PointPatchMix: Point Cloud Mixing with Patch Scoring

# Introduction
We propose PointPatchMix, a novel point cloud mixing method that operates at the patch level and integrates a patch scoring module. You can check out [paper](https://arxiv.org/abs/2303.06678) for more details.
!(https://github.com/winfred2027/PointPatchMix/blob/main/figures/main.png)

# Installation
PyTorch >= 1.7.0 < 1.11.0; python >= 3.7; CUDA >= 9.0; GCC >= 4.9; torchvision;
```
pip install -r requirements.txt
```
```
# Chamfer Distance & emd
cd ./emd
TORCH_CUDA_ARCH_LIST="3.5;5.0;6.0;6.1;7.0;7.5+PTX" python setup.py install --user
# PointNet++
pip install "git+https://github.com/erikwijmans/Pointnet2_PyTorch.git#egg=pointnet2_ops&subdirectory=pointnet2_ops_lib"
# GPU kNN
pip install --upgrade https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl
```

# DataSets


# Training
```
CUDA_VISIBLE_DEVICES=<GPU> python main.py --config cfgs/pretrain.yaml --exp_name <output_file_name>
```

# Acknowledgements
Our codes are built upon [Point-MAE](https://github.com/Pang-Yatian/Point-MAE) and [PointCutMix](https://github.com/cuge1995/PointCutMix)

# Citation
If you find our code helpful, please cite our paper:
```
@article{wang2023pointpatchmix,
  title={PointPatchMix: Point Cloud Mixing with Patch Scoring},
  author={Wang, Yi and Wang, Jiaze and Li, Jinpeng and Zhao, Zixu and Chen, Guangyong and Liu, Anfeng and Heng, Pheng-Ann},
  journal={arXiv preprint arXiv:2303.06678},
  year={2023}
}
```
