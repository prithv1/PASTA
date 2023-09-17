# PASTA

This repository provides code for our paper:
 
[PASTA: Proportional Amplitude Spectrum Training Augmentation for Syn-to-Real Domain Generalization](https://arxiv.org/abs/2212.00979)

[Prithvijit Chattopadhyay*](https://prithv1.xyz/), [Kartik Sarangmath*](https://www.semanticscholar.org/author/Kartik-Sarangmath/2046863986), [Vivek Vijaykumar](https://www.linkedin.com/in/vivek-vjk), [Judy Hoffman](https://faculty.cc.gatech.edu/~judy/)

(*equal contribution)

ICCV 2023

 ### Contents

<div class="toc">
<ul>
<li><a href="#-pasta">📝 What is PASTA?</a></li>
<li><a href="#-setup">💻 Setup</a></li>
<li><a href="#-experiments">📊 Experiments</a></li>
<li><a href="#-checkpoints">📊 Checkpoints</a></li>
</ul>
</div>

## 📝 What is PASTA?

<img src="media/pasta.png" alt="" width="100%">

Synthetic data offers the promise of cheap and bountiful training data for settings where lots of labeled real-world data for tasks is unavailable. However, models trained on synthetic data significantly underperform on real-world data. In this paper, we propose Proportional Amplitude Spectrum Training Augmentation (PASTA),a simple and effective augmentation strategy to improve out-of-the-box synthetic-to-real (syn-to-real) generalization performance. PASTA involves perturbing the amplitude spectrums of the synthetic images in the Fourier domain to generate augmented views. We design PASTA to perturb the amplitude spectrums in a structured manner such that high-frequency components are perturbed relatively more thanthe low-frequency ones. For the tasks of semantic segmentation (GTAV→Real),object detection (Sim10K→Real), and object recognition (VisDA-C Syn→Real), across a total of 5 syn-to-real shifts, we find that PASTA outperforms more complexstate-of-the-art generalization methods while being complementary to the same.

To visualize PASTA augmented samples, follow the steps below:

1. First download and setup [Anaconda](https://docs.anaconda.com/anaconda/install/linux/) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html).
2. Create and activate a conda (anaconda / miniconda) environment
```
conda create -n pasta python=3.8
conda activate pasta
```
3. Install dependencies
```
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 -c pytorch
pip install jupyter
```
4. Play around with PASTA [here](https://github.com/prithv1/PASTA/blob/main/pasta.ipynb)

## 💻 Setup

We conduct experiments on semantic segmentation, object detection and object recognition and build on top of the following repositories to do so - <a href="https://github.com/shachoi/RobustNet">RobustNet (SemSeg)</a>, <a href="https://github.com/open-mmlab/mmdetection">mmdetection (ObjDet)</a>, <a href="https://github.com/NVlabs/CSG">CSG (ObjRecog)</a>.

#### Semantic Segmentation

Follow instructions under <a href="https://github.com/ksarangmath/PASTA_robustnet">RobustNet</a> to download datasets and install dependencies required to run experiments for semantic segmentation.

Once downloaded, update the path to respective datasets in the [config](https://github.com/ksarangmath/PASTA_robustnet/blob/049c59908c1b10cc504d89d5183eb018d810267d/config.py).

#### Object Detection

<!-- Follow instructions under <a href="https://github.com/prithv1/mmdetection">mmdetection</a> install dependencies required to run experiments for object recognition. -->

Follow <a href="https://github.com/prithv1/mmdetection/blob/master/docs/en/get_started.md/#Installation">these instructions</a> to install dependencies and setup [mmdetection](https://github.com/prithv1/mmdetection).

Download the <a href="https://fcav.engin.umich.edu/projects/driving-in-the-matrix">Sim10k</a> dataset and run the following command to process annotations.

```
python dataset_utils/sim10k_voc2coco_format.py \
    --sim10k_path <path-to-sim10k-folder> \
    --img-dir <path-to-sim10k-images> \
    --gt-dir <path-to-sim10k-annotations> \
    --out-dir <path-to-store-processed-annotations>
```

Download the <a href="https://www.cityscapes-dataset.com/downloads/">Cityscapes</a> dataset.

Once processed, update the path to individual datasets in the experiment [configs](https://github.com/prithv1/mmdetection/tree/74f883ef796e1bea3c5ceea41e4bc826486e8d0c/configs/pasta_dg).


#### Object Recognition

<!-- Follow instructions under [CSG](CSG) to download datasets and install dependencies required to run experiments for object recognition. -->

Follow <a href="https://github.com/vivekvjk/PASTA_CSG/blob/c7a75d5c63db1a60b28e300457da74bb428073bd/README.md">these instructions</a> to download the VisDA-C dataset.

Once downloaded, update the default path to datasets in the [training script](https://github.com/vivekvjk/PASTA_CSG/blob/c7a75d5c63db1a60b28e300457da74bb428073bd/train.py#L31).

To install required dependencies, follow the steps below:
1. First download and setup [Anaconda](https://docs.anaconda.com/anaconda/install/linux/) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html).
2. Create and activate a conda (anaconda / miniconda) environment
```
conda create -n csg python=3.8
conda activate csg
```
3. Install dependencies
```
pip install requirements.txt
```

## 📊 Experiments

#### Semantic Segmentation

To run semantic segmentation experiments with PASTA, navigate to [PASTA_robustnet](https://github.com/ksarangmath/PASTA_robustnet) and run the following commands.

1. MobileNetv2 Backbone (trained on 2 GPUs)
```
# Train: GTAV, Test: BDD100K, Cityscapes, Synthia, Mapillary / MobileNetV2, Baseline + PASTA
CUDA_VISIBLE_DEVICES=0,1 ./scripts/PASTA/train_mobile_gtav_base_PASTA.sh 

# Train: GTAV, Test: BDD100K, Cityscapes, Synthia, Mapillary / MobileNetV2, IBN-Net + PASTA
CUDA_VISIBLE_DEVICES=0,1 ./scripts/PASTA/train_mobile_gtav_ibn_PASTA.sh

# Train: GTAV, Test: BDD100K, Cityscapes, Synthia, Mapillary / MobileNetV2, ISW + PASTA
CUDA_VISIBLE_DEVICES=0,1,2,3 ./scripts/PASTA/train_mobile_gtav_isw_PASTA.sh
```

2. ResNet-50 Backbone (trained on 4 GPUs)
```
# Train: GTAV, Test: BDD100K, Cityscapes, Synthia, Mapillary / ResNet50, Baseline + PASTA
CUDA_VISIBLE_DEVICES=0,1,2,3 ./scripts/PASTA/train_r50os16_gtav_base_PASTA.sh

# Train: GTAV, Test: BDD100K, Cityscapes, Synthia, Mapillary / ResNet50, IBN-Net + PASTA
CUDA_VISIBLE_DEVICES=0,1,2,3 ./scripts/PASTA/train_r50os16_gtav_ibn_PASTA.sh

# Train: GTAV, Test: BDD100K, Cityscapes, Synthia, Mapillary / ResNet50, ISW + PASTA
CUDA_VISIBLE_DEVICES=0,1,2,3 ./scripts/PASTA/train_r50os16_gtav_isw_PASTA.sh
```

3. ResNet-101 Backbone (trained on 4 GPUs, atleast 24G VRAM)

```
# Train: GTAV, Test: BDD100K, Cityscapes, Synthia, Mapillary / ResNet101, Baseline + PASTA
CUDA_VISIBLE_DEVICES=0,1,2,3 ./scripts/PASTA/train_r101os8_gtav_base_PASTA.sh

# Train: GTAV, Test: BDD100K, Cityscapes, Synthia, Mapillary / ResNet101, IBN-Net + PASTA
CUDA_VISIBLE_DEVICES=0,1,2,3 ./scripts/PASTA/train_r101os8_gtav_ibn_PASTA.sh

# Train: GTAV, Test: BDD100K, Cityscapes, Synthia, Mapillary / ResNet101, ISW + PASTA
CUDA_VISIBLE_DEVICES=0,1,2,3 ./scripts/PASTA/train_r101os8_gtav_isw_PASTA.sh
```

All models are trained for 40k iterations. Once trained, obtain syn-to-real generalization performance by:
1. Finding the best in-domain checkpoint epoch from the experiment directory
2. Picking results for the corresponding epoch from the training logs

#### Object Detection

To run object detection experiments with PASTA, navigate to [PASTA_mmdetection](https://github.com/prithv1/mmdetection) and run the following commands.

1. ResNet-50 Backbone (trained on 4 GPUs)
```
# Baseline Faster-RCNN
./tools/dist_train.sh configs/pasta_dg/vanilla_faster_rcnn_r50_sim10k_detection_dg.py 4

# Baseline Faster-RCNN (with Photometric Distortion)
./tools/dist_train.sh configs/pasta_dg/vanilla_faster_rcnn_r50_sim10k_detection_dg_pd.py 4

# Baseline Faster-RCNN (with PASTA)
./tools/dist_train.sh configs/pasta_dg/vanilla_faster_rcnn_r50_sim10k_detection_dg_pasta.py 4

# Baseline Faster-RCNN (with PASTA + Photometric Distortion)
./tools/dist_train.sh configs/pasta_dg/vanilla_faster_rcnn_r50_sim10k_detection_dg_pasta_pd.py 4
```

2. ResNet-101 Backbone (trained on 4 GPUs)

```
# Baseline Faster-RCNN
./tools/dist_train.sh configs/pasta_dg/vanilla_faster_rcnn_r101_sim10k_detection_dg.py 4

# Baseline Faster-RCNN (with Photometric Distortion)
./tools/dist_train.sh configs/pasta_dg/vanilla_faster_rcnn_r101_sim10k_detection_dg_pd.py 4

# Baseline Faster-RCNN (with PASTA)
./tools/dist_train.sh configs/pasta_dg/vanilla_faster_rcnn_r101_sim10k_detection_dg_pasta.py 4

# Baseline Faster-RCNN (with PASTA + Photometric Distortion)
./tools/dist_train.sh configs/pasta_dg/vanilla_faster_rcnn_r101_sim10k_detection_dg_pasta_pd.py 4
```

All models are trained for 10k iterations. Once trained, obtain syn-to-real generalization performance at 10k iters from the respective log files.

#### Object Recognition

To run object recognition experiments with PASTA, navigate to [PASTA_CSG](https://github.com/vivekvjk/PASTA_CSG) and run the following commands.

ResNet-101 Backbone
```
# Train Baseline
./scripts/Visda/train.sh

# Train Baseline + PASTA
./scripts/Visda/train_PASTA.sh
```

To evaluate trained models, run the following commands.
```
# Evaluate Baseline
./scripts/Visda/eval.sh

# Evaluate Baseline + PASTA
./scripts/Visda/eval_PASTA.sh
```

## 📊 Checkpoints

[Coming Soon]

## Contributors

* [Prithvijit Chattopadhyay](https://github.com/prithv1)
* [Kartik Sarangmath](https://github.com/ksarangmath)
* [Vivek Vijaykumar](https://github.com/vivekvjk)

## Cite

Please cite our work if you find it useful:
```
@inproceedings{2023iccv_PASTA, 
    author = {Chattopadhyay*, Prithvijit and Sarangmath*, Kartik and Vijaykumar, Vivek and Hoffman, Judy},
    title = {PASTA: Proportional Amplitude Spectrum Training Augmentation for Syn-to-Real Domain Generalization},
    year = 2023,
    booktitle = {IEEE/CVF International Conference in Computer Vision (ICCV)}
}
```
