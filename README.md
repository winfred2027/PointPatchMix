# PointPatchMix: Point Cloud Mixing with Patch Scoring
We propose PointPatchMix, a novel point cloud mixing method that operates at the patch level and integrates a patch scoring module. You can check out [paper](https://arxiv.org/abs/2303.06678) for more details.
![](https://github.com/winfred2027/PointPatchMix/blob/main/figures/main.png)

**The overall scheme of the PointPatchMix.** (a) The original point clouds are divided into multiple patches, subsequently undergoing mask token processing and mixing. (b) A pre-trained teacher model assigns each patch with a content-based significance score. The ground truth of the mixed point cloud is ascertained by aggregating the scores of designated patches.

# Installation
PyTorch >= 1.7.0 < 1.11.0; python >= 3.7; CUDA >= 9.0; GCC >= 4.9; torchvision;
```
pip install -r requirements.txt
```
```
# EMD
cd ./emd
TORCH_CUDA_ARCH_LIST="3.5;5.0;6.0;6.1;7.0;7.5+PTX" python setup.py install --user
# PointNet++
pip install "git+https://github.com/erikwijmans/Pointnet2_PyTorch.git#egg=pointnet2_ops&subdirectory=pointnet2_ops_lib"
# GPU kNN
pip install --upgrade https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl
```

# Datasets
We conduct extensive experiments on both synthetic and real-world datasets in point cloud shape classification to evaluate the effectiveness of PointPatchMix, i.e., ModelNet40 and ScanObjetNN. 

Please download the datasets from [DATASET.md in Point-MAE](https://github.com/Pang-Yatian/Point-MAE/blob/main/DATASET.md). 

Then, we adopt Point-MAE as our teacher model to generate the patch token scores and save the scores offline to improve training efficiency.

# Experiments
Train Point-MAE on ModelNet40, run:
```
CUDA_VISIBLE_DEVICES=<GPU> python main.py --config cfgs/patchmix_modelnet40_pointmae.yaml \
--scratch_model --score --exp_name <output_file_name>
```
Train Point-MAE on ScanObjectNN, run:
```
CUDA_VISIBLE_DEVICES=<GPU> python main.py --config cfgs/patchmix_scanobjectnnhard_pointmae.yaml \
--scratch_model --score --exp_name <output_file_name>
```
Finetune Point-MAE on ModelNet40, run:
```
CUDA_VISIBLE_DEVICES=<GPU> python main.py --config cfgs/patchmix_modelnet40_pointmae.yaml \
--finetune_model --score --exp_name <output_file_name> --ckpts <path/to/pre-trained/model>
```
If you want to try different models or datasets, first create a new config file, and pass its path to --config.
```
CUDA_VISIBLE_DEVICES=<GPU> python main.py --config cfgs/<config_file> \
--scratch_model --score --exp_name <output_file_name>
```

# Acknowledgements
Our codes are built upon [Point-MAE](https://github.com/Pang-Yatian/Point-MAE) and [PointCutMix](https://github.com/cuge1995/PointCutMix).

# Citation
If you find our code helpful, please cite our paper:
```
@inproceedings{wang2024pointpatchmix,
  title={PointPatchMix: Point Cloud Mixing with Patch Scoring},
  author={Wang, Yi and Wang, Jiaze and Li, Jinpeng and Zhao, Zixu and Chen, Guangyong and Liu, Anfeng and Heng, Pheng-Ann},
  journal={Thirty-Eighth AAAI Conference on Artificial Intelligence (AAAI-24)},
  year={2024}
}
```
