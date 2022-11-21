# PASTA

PyTorch code for our paper: <a href="https://arxiv.org/abs/2106.04531">PASTA: Proportional Amplitude Spectrum Augmentation for Synthetic to Real Domain Generalization</a>.

Prithvijit Chattopadhyay*, Kartik Sarangmath*, Vivek Vijaykumar, Judy Hoffman

(*equal contribution)

 <!-- _Synthetic data offers the promise of cheap and bountiful training data for settings where lots of labeled real-world data for some task is unavailable. However, models trained on synthetic data significantly underperform on real-world data. In this paper, we propose Proportional Amplitude Spectrum Training Augmentation (PASTA), a simple and effective augmentation strategy to improve out-of-the-box synthetic-to-real (syn-to-real) generalization performance. PASTA involves perturbing the amplitude spectrums of the synthetic images in the Fourier domain to generate augmented views. We design PASTA to perturb the amplitude spectrums in a structured manner such that high-frequency components are perturbed relatively more than the low-frequency ones. For the tasks of semantic segmentation (GTAV→Real), object detection (Sim10K→Real), and object recognition (VisDA-C Syn→Real), across a total of 5 syn-to-real shifts, we find that PASTA  either outperforms or is consistently competitive with more complex state-of-the-art methods while being complementary to other generalization approaches._ -->

 ### Contents

<div class="toc">
<ul>
<li><a href="#-pasta">📝 What is PASTA?</a></li>
<li><a href="#-setup">💻 Setup</a></li>
<li><a href="#-experiments">📊 Experiments</a></li>
</ul>
</div>

## 📝 What is PASTA?

<img src="media/pasta.png" alt="" width="100%">

PASTA is a simple and effective frequency domain augmentation strategy to improve out-of-the-box synthetic-to-real (syn-to-real) generalization performance. PASTA involves perturbing the amplitude spectra of the synthetic images in the Fourier domain to generate augmented views. We find that synthetic images tend to be less diverse in their high-frequency components compared to real ones. Based on this observation, we design PASTA to perturb the amplitude spectrums in a structured manner such that high-frequency components are perturbed relatively more than the low-frequency ones (as outlined in the figure above). For the tasks of semantic segmentation (GTAV→Real), object detection (Sim10K→Real), and object recognition (VisDA-C Syn→Real), across a total of 5 syn-to-real shifts, we find that PASTA  either outperforms or is consistently competitive with more complex state-of-the-art methods while being complementary to other generalization approaches.

## 💻 Setup

We conduct experiments on semantic segmentation, object detection and object recognition and build on top of the following repositories to do so - <a href="https://github.com/shachoi/RobustNet">RobustNet (SemSeg)</a>, <a href="https://github.com/open-mmlab/mmdetection">mmdetection (ObjDet)</a>, <a href="https://github.com/NVlabs/CSG">CSG (ObjRecog)</a>.

#### Semantic Segmentation

Follow instructions under [Robustnet](RobustNet) to download datasets and install dependencies required to run experiments for semantic segmentation.

#### Object Detection

Follow instructions under <a href="https://github.com/prithv1/mmdetection">mmdetection</a> install dependencies required to run experiments for object recognition.

Download the <a href="https://fcav.engin.umich.edu/projects/driving-in-the-matrix">Sim10k</a> dataset. Once downloaded run the following command to process annotations.

```
python dataset_utils/sim10k_voc2coco_format.py \
    --sim10k_path <path-to-sim10k-folder> \
    --img-dir <path-to-sim10k-images> \
    --gt-dir <path-to-sim10k-annotations> \
    --out-dir <path-to-store-processed-annotations>
```


#### Object Recognition

Follow instructions under [CSG](CSG) to download datasets and install dependencies required to run experiments for object recognition.

## 📊 Experiments
