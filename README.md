# <a href="https://arxiv.org/abs/2106.04531">PASTA: Proportional Amplitude Spectrum Augmentation for Synthetic to Real Domain Generalization</a>

#### Prithvijit Chattopadhyay*, Kartik Sarangmath*, Vivek Vijaykumar, Judy Hoffman

(*equal contribution)

<img src="media/pasta.png" alt="" width="100%">

 _Synthetic data offers the promise of cheap and bountiful training data for settings where lots of labeled real-world data for some task is unavailable. However, models trained on synthetic data significantly underperform on real-world data. In this paper, we propose Proportional Amplitude Spectrum Training Augmentation (PASTA), a simple and effective augmentation strategy to improve out-of-the-box synthetic-to-real (syn-to-real) generalization performance. PASTA involves perturbing the amplitude spectrums of the synthetic images in the Fourier domain to generate augmented views. We design PASTA to perturb the amplitude spectrums in a structured manner such that high-frequency components are perturbed relatively more than the low-frequency ones. For the tasks of semantic segmentation (GTAV→Real), object detection (Sim10K→Real), and object recognition (VisDAC Syn→Real), across a total of 5 syn-to-real shifts, we find that PASTA  either outperforms or is consistently competitive with more complex state-of-the-art methods while being complementary to other generalization approaches._

 ### Contents

<div class="toc">
<ul>
<li><a href="#-installation">💻 Installation</a></li>
<li><a href="#-pasta">📝 PASTA</a></li>
<!-- <li><a href="#-dataset">📊 Dataset</a></li> -->
<li><a href="#-segmentation-experiments">🖼️ Semantic Segmentation</a></li>
<li><a href="#-detection-experiments">🖼️ Object Detection</a></li>
<li><a href="#-recognition-experiments">🖼️ Object Recognition</a></li>
</ul>
</div>

## 💻 Installation & 🖼️ Datasets

We conduct experiments on semantic segmentation, object detection and object recognition and build on top of the following repositories to do so - <a href="https://github.com/shachoi/RobustNet">RobustNet (SemSeg)</a>, <a href="https://github.com/open-mmlab/mmdetection">mmdetection (ObjDet)</a>, <a href="https://github.com/NVlabs/CSG">CSG (ObjRecog)</a>.

#### Semantic Segmentation

Follow instructions under the folder [Robustnet](RobustNet) to download datasets and install dependencies required to run experiments for semantic segmentation.

#### Object Detection

Follow instructions under the folder [mmdetection](mmdetection) install dependencies required to run experiments for object recognition.

Download the <a href="https://fcav.engin.umich.edu/projects/driving-in-the-matrix">Sim10k</a> dataset. Once downloaded run the following command to process annotations.

```
python dataset_utils/sim10k_voc2coco_format.py \
    --sim10k_path <path-to-sim10k-folder> \
    --img-dir <path-to-sim10k-images> \
    --gt-dir <path-to-sim10k-annotations> \
    --out-dir <path-to-store-processed-annotations>
```


#### Object Recognition

Follow instructions under the [CSG](CSG) to download datasets and install dependencies required to run experiments for object recognition.

## 📝 PASTA
