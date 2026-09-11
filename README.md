# AA-ViT
Official PyTorch Implementation of **AA-ViT (Anatomically Aware Vision Transformer)**, a model for contrast-enhanced brain MRI synthesis built on top of the **ResViT** baseline. AA-ViT is described in our MIUA 2026 (METIS Workshop) paper:

O. T. Meraj, T. Flannery, C. Cummins, M. Townend, T. C. Booth, P. Crossley, M. McCann, I. Overton and S. Unnikrishnan, "AA-ViT: Anatomically Aware Vision Transformer with Structural and Frequency Guidance for Contrast Enhanced Brain MRI Synthesis," accepted at MIUA 2026, METIS Workshop. Preprint: [arXiv:2607.07553](https://arxiv.org/abs/2607.07553).

> This is a preprint citation — it will be updated to the camera-ready conference proceedings once published.

<img src="aa_vit_architecture.png" width="800px"/>

*Overview of the proposed AA-ViT framework. (A) Generator–discriminator architecture for synthesizing contrast-enhanced MRI (CEMRI) from pre-contrast MRI inputs. (B) Residual Dense Edge Block (RDEB) for gradient-based edge feature representation. (C) Multi-component anatomically aware training objective.*

## Dependencies

```
python>=3.6.9
torch>=1.7.1
torchvision>=0.8.2
visdom
dominate
scikit-image
h5py
scipy
ml_collections
cuda=>11.2
```
## Installation
- Clone this repo:
```bash
git clone https://github.com/TalhaMeraj/AA-ViT
cd AA-ViT
```

## Download pre-trained ViT models from Google
* [Pre-trained ViT models](https://console.cloud.google.com/storage/vit_models/):
```bash
wget https://storage.googleapis.com/vit_models/imagenet21k/R50+ViT-B_16.npz &&
mkdir ../model/vit_checkpoint/imagenet21k &&
mv {MODEL_NAME}.npz ../model/vit_checkpoint/imagenet21k/R50-ViT-B_16.npz
```

## Dataset
To reproduce the results reported in the paper, we recommend the following dataset processing steps:

Sequentially select subjects from the dataset.
Apply skull-stripping to 3D volumes.
Select 2D cross-sections from each subject.
Normalize the selected 2D cross-sections before training and before metric calculation.

Unlike the original ResViT data pipeline (2 modalities packed into the R/G channels of an RGB image), this codebase's loader ([data/aligned_dataset.py](data/aligned_dataset.py)) reads a single **4-channel `.tiff`** file per slice, with channels in the fixed order **[T1, T2, FLAIR, T1ce]**. For `--which_direction AtoB`, the input `A` is `[T1, T2, FLAIR]` (3-channel, `--input_nc 3`) and the target `B` is `T1ce` (1-channel, `--output_nc 1`); `--which_direction BtoA` reverses this.

You should structure your aligned dataset in the following way:
```
/Datasets/BraTS2021/T1_T2_FLAIR_T1ce/
  ├── train
  ├── val
  ├── test
```
Each file under `train`/`val`/`test` is a single co-registered, 4-channel `.tiff` slice (one channel per modality, in the order above).

## Pre-training of ART blocks without the presence of transformers
It is recommended to pretrain the convolutional parts of the model before inserting transformer modules and fine-tuning. This significantly improves training stability.

```
python3 train.py --dataroot Datasets/BraTS2021/T1_T2_FLAIR_T1ce/ --name T1_T2_Flair_T1ce_nc_3_pretrained_batch_32_AAViT --gpu_ids 0 --model resvit_many --which_model_netG res_cnn 
--which_direction AtoB --lambda_A 100 --dataset_mode aligned --norm batch --pool_size 0 --output_nc 1 --input_nc 3 --loadSize 256 --fineSize 256 
--niter 50 --niter_decay 50 --save_epoch_freq 5 --checkpoints_dir checkpoints/ --display_id 0 --lr 0.0002
```

<br />
<br />

## Fine tune AA-ViT
```
python3 train.py --dataroot Datasets/BraTS2021/T1_T2_FLAIR_T1ce/ --name T1_T2_Flair_T1ce_nc_3_L1Tune_EMA_diffmap_RDEB_Edge_batch_4_AAViT --gpu_ids 0 --model resvit_many --which_model_netG resvit 
--which_direction AtoB --lambda_A 100 --dataset_mode aligned --norm batch --pool_size 0 --output_nc 1 --input_nc 3 --loadSize 256 --fineSize 256 
--niter 50 --niter_decay 50 --save_epoch_freq 50 --checkpoints_dir checkpoints/ --display_id 0 --batchSize 4 
--pre_trained_transformer 1 --pre_trained_resnet 1 --pre_trained_path checkpoints/T1_T2_Flair_T1ce_nc_3_pretrained_batch_32_AAViT/latest_net_G.pth --lr 0.0001 
--lambda_diffmap 10.0 --lambda_edge 10.0 --lambda_fft 0.05 --lambda_ms 10.0
```
`--lambda_diffmap`, `--lambda_edge`, `--lambda_fft` and `--lambda_ms` weight AA-ViT's anatomically aware loss terms (difference-map, edge, frequency and multi-scale consistency); the Residual Dense Edge Block (RDEB) itself is always part of the generator, not a flag. An exponential moving average (EMA) copy of the generator is kept by default during training — disable it with `--no_ema`, or tune `--ema_decay_g` (default `0.999`).

<br />
<br />

## Testing
```
python3 test.py --dataroot Datasets/BraTS2021/T1_T2_FLAIR_T1ce/ --name T1_T2_Flair_T1ce_nc_3_L1Tune_EMA_diffmap_RDEB_Edge_batch_4_AAViT --gpu_ids 0 --model resvit_many --which_model_netG resvit 
--dataset_mode aligned --norm batch --phase test --output_nc 1 --input_nc 3 --how_many 10000 --serial_batches --fineSize 256 --loadSize 256 
--results_dir results/ --checkpoints_dir checkpoints/ --which_epoch latest
```
Testing loads the EMA generator checkpoint by default; pass `--no_ema` to evaluate the raw (non-EMA) weights instead.

# Citation
You are encouraged to modify/distribute this code. However, please acknowledge this code and cite the AA-ViT paper appropriately.
```
@misc{meraj2026aavit,
  title={AA-ViT: Anatomically Aware Vision Transformer with Structural and Frequency Guidance for Contrast Enhanced Brain MRI Synthesis},
  author={Meraj, Talha and Flannery, Tom and Cummins, Charlie and Townend, Matt and Booth, Thomas C. and Crossley, Peter and McCann, Michael and Overton, Ian and Unnikrishnan, Saritha},
  year={2026},
  eprint={2607.07553},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  note={Accepted at MIUA 2026, METIS Workshop. Preprint -- will be updated to the conference proceedings citation once published.},
  url={https://arxiv.org/abs/2607.07553}
}
```
For any questions, comments and contributions, please contact Talha Meraj (talhameraj32[at]gmail.com). <br />

(c) 2026 Talha Meraj et al.

## Acknowledgments
This code builds on [ResViT](https://github.com/icon-lab/ResViT) (Dalmaz et al., IEEE TMI 2022, [paper](https://ieeexplore.ieee.org/document/9758823)), which itself uses libraries from [pGAN](https://github.com/icon-lab/pGAN-cGAN) and [pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) repository.
