<div align="center">

# Mobile-CoHAtNet

### A Lightweight Multimodal CNN–Transformer for Efficient Camera Localization from RGB-D and IMU Data

[Hussein Hasan](https://orcid.org/0009-0002-8486-0771)<sup>1</sup> · Miguel Angel Garcia<sup>2</sup> · Hatem Rashwan<sup>1</sup> · Domenec Puig<sup>1</sup>

<sup>1</sup> University Rovira i Virgili, Tarragona, Spain &nbsp;&nbsp; <sup>2</sup> Autonomous University of Madrid, Madrid, Spain

*Machine Vision and Applications* · Vol. 37 · Article 171 · 2026

[![Paper](https://img.shields.io/badge/Paper-Springer-0A66C2)](https://link.springer.com/article/10.1007/s00138-026-01935-5)
[![Full Text](https://img.shields.io/badge/Full%20Text-Free%20Read-E37400)](https://rdcu.be/JYbNfJXiSTsa)
[![Dataset](https://img.shields.io/badge/Dataset-Google%20Drive-34A853?logo=googledrive&logoColor=white)](https://drive.google.com/drive/folders/1DRH1vohn71Mv8_6adYFZVhAspAv2nuRF?usp=sharing)

<a href="https://deepwiki.com/Husseinhhameed/Mobile-CoHAtNet"><img src="https://img.shields.io/badge/Ask-DeepWiki-7C3AED?style=for-the-badge&labelColor=1E1B4B" alt="Ask DeepWiki" height="38"></a>

[📖 Overview](#-overview) · [🧩 Architecture](#-architecture) · [🔥 Attention](#-attention-visualization) · [📱 Dataset](#-self-collected-mobile-dataset) · [📝 Citation](#-citation)

</div>

---

## 📖 Overview

Camera localization, i.e. recovering a camera's 6-DoF pose in 3D space, is a key building block of AR/VR, autonomous navigation and robotics. Existing learning-based localizers struggle to be both accurate and efficient: convolutional networks are fast and good at local geometry but miss the global picture, Transformers see the whole scene but are expensive to run, and a single RGB camera cannot resolve metric scale on its own, which limits pose precision.

**Mobile-CoHAtNet** is a multimodal hybrid CNN–Transformer built with efficiency as the first priority. With only **4.23 M parameters (16.9 MB)**, it combines RGB, depth and inertial (IMU) data in one compact network through three design choices:

- 🔗 **Geometry-aware hybrid attention:** MBConv features are injected into the *Value* branch of self-attention, so local and global reasoning happen inside the same lightweight block.
- 🎨 **Unified RGB-D backbone:** a single backbone processes colour and depth together, replacing the usual dual-encoder design and its redundant feature extractors.
- 🧭 **Lightweight late IMU fusion:** a small inertial module refines the pose estimate at negligible extra cost.

On **7-Scenes** and **Cambridge Landmarks**, Mobile-CoHAtNet performs on par with recent state-of-the-art methods, and better in several cases, despite being much smaller, and it runs at **73.5 FPS on an off-the-shelf smartphone**. Real-world experiments with smartphone RGB-D frames and measured IMU signals further show that it stays robust under difficult visual and geometric conditions. The takeaway: a large backbone is not a prerequisite for competitive 6-DoF pose regression, and a design built around efficiency can run on resource-limited hardware.

| 🪶 Parameters | 💾 Model Size | ⚡ Speed | 🧪 Benchmarks |
|:---:|:---:|:---:|:---:|
| **4.23 M** | **16.9 MB** | **73.5 FPS** on a smartphone | 7-Scenes · Cambridge Landmarks |

> [!TIP]
> 📄 **Read the full paper for free:** [rdcu.be/JYbNfJXiSTsa](https://rdcu.be/JYbNfJXiSTsa)

## 🧩 Architecture

<p align="center">
  <img src="model.png" alt="Mobile-CoHAtNet architecture" width="95%">
</p>
<p align="center"><em>Architecture of Mobile-CoHAtNet.</em></p>

> [!NOTE]
> The full model implementation is provided in [`Mobile-CoHAtNet.py`](Mobile-CoHAtNet.py).

## 🔥 Attention Visualization

<p align="center">
  <img src="attention.png" alt="Attention heatmaps of Mobile-CoHAtNet" width="95%">
</p>
<p align="center"><em>Attention heatmaps of Mobile-CoHAtNet.</em></p>

## 📱 Self-Collected Mobile Dataset

For the real-world evaluation, we recorded a laboratory sequence with an off-the-shelf **iPhone 14 Pro Max** using the [Stray Scanner](https://github.com/strayrobots/scanner) app. The sequence contains synchronized RGB frames, LiDAR depth and real IMU measurements, and the scene includes cluttered areas, texture-less surfaces and repetitive patterns that make localization challenging.

<p align="center"><b>📷 RGB</b> &nbsp;•&nbsp; <b>📏 LiDAR Depth</b> &nbsp;•&nbsp; <b>🧭 Real IMU</b> &nbsp;•&nbsp; <b>📱 iPhone 14 Pro Max</b></p>

<p align="center">📥 <b><a href="https://drive.google.com/drive/folders/1DRH1vohn71Mv8_6adYFZVhAspAv2nuRF?usp=sharing">Download the dataset from Google Drive</a></b></p>

<p align="center">
  <img src="dataset.png" alt="Sample from the self-collected mobile dataset" width="95%">
</p>
<p align="center"><em>Sample from the self-collected mobile dataset.</em></p>

## 📝 Citation

If you find this work useful, please cite:

> Hasan, H., Garcia, M.A., Rashwan, H. *et al.* Mobile-CoHAtNet: a lightweight multimodal CNN–transformer for efficient camera localization from RGB-D and IMU data. *Machine Vision and Applications* **37**, 171 (2026). https://doi.org/10.1007/s00138-026-01935-5

```bibtex
@article{hasan2026mobilecohatnet,
  author    = {Hasan, Hussein and Garcia, Miguel Angel and Rashwan, Hatem and Puig, Domenec},
  title     = {{Mobile-CoHAtNet}: a lightweight multimodal {CNN}--transformer for efficient camera localization from {RGB-D} and {IMU} data},
  journal   = {Machine Vision and Applications},
  volume    = {37},
  number    = {6},
  pages     = {171},
  year      = {2026},
  publisher = {Springer},
  doi       = {10.1007/s00138-026-01935-5},
  url       = {https://doi.org/10.1007/s00138-026-01935-5}
}
```

---

<div align="center">

⭐ **If you find this work helpful, please consider giving the repository a star!**

</div>
