# ODPG-Outfitting-Diffusion-with-Pose-Guided-conditions🚀 & Accepted at CKAIA 2024📈

<div align="center">

[![CKAIA 2024](https://img.shields.io/badge/CKAIA%202024-Accepted-brightgreen)](YourConferenceLinkHere)&nbsp;
[![Poster](https://img.shields.io/badge/Demo-Try%20it%20out-blue)](YourDemoLinkHere)&nbsp;
[![GitHub stars](https://img.shields.io/github/stars/seohyun8825/ODPG_1?style=social)](https://github.com/seohyun8825/ODPG_1)

</div>

<p align="center">
  <strong>Our model adapts to both pose and garment features, providing a high-quality virtual try-on experience.</strong>
  
</p>

<p align="center">
  <img src="https://github.com/seohyun8825/ODPG_1/assets/153355118/da31df0c-4179-4a6e-a280-2500d0d003c9" width=95%>
</p>

## 📜 Model Pipeline
Below is the pipeline of our model, detailing each step from input to output:

![pipeline_ODPG](https://github.com/seohyun8825/ODPG_1/assets/153355118/a4e2c20e-5a0c-4ab8-b9ea-c5de18c64d9e)

## 📊 Qualitative Results
Here are the results after 10 hours of training (20 epochs) using a single A100 GPU:

![qualitative result](https://github.com/seohyun8825/ODPG_1/assets/153355118/da31df0c-4179-4a6e-a280-2500d0d003c9)

## 🗂️ Dataset Preparation
To prepare the "In-shop Clothes Retrieval Benchmark" dataset, follow these steps:

1. Download the dataset from [DeepFashion: In-shop Clothes Retrieval Benchmark](https://mmlab.ie.cuhk.edu.hk/projects/DeepFashion/InShopRetrieval.html).
2. This dataset includes:
   - 7,982 clothing items.
   - 52,712 in-shop clothes images.
   - Approximately 200,000 cross-pose/scale pairs.
   - Each image is annotated with bounding box, clothing type, and pose type.
3. Extract the downloaded files into the `Fashion` folder within your project directory to maintain the required structure.

## 📋 TODO

- [ ] Checkpoint update
- [ ] Training scripts with detailed usage instructions
- [ ] Scripts for ablation studies
- [x] ~~Model pipeline~~

## 🔧 Installation
To set up and run our model, follow these steps:

1. Clone the repository: `git clone https://github.com/seohyun8825/ODPG_1.git`
2. Install required packages: `pip install -r requirements.txt`

## 🔍 Code Base and Modifications
This project is built on top of the [CFLD official code](https://github.com/YanzuoLu/CFLD). The original codebase has been significantly modified to include additional conditioning on garment features, enabling the model to handle more complex virtual try-on scenarios where both pose and clothing attributes are considered.

