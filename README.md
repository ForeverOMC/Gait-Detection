---
typora-root-url: images
---

# Gait Detection

## Introduction

<img src="/../README.assets/video_animation-1734873642526.gif" alt="video_animation" style="zoom:50%;" />

<img src="/../README.assets/contact_sequence copy.png" alt="contact_sequence copy" style="zoom: 67%;" />

<img src="/../README.assets/video_animation copy-1734873766390.gif" alt="video_animation copy" style="zoom:50%;" />

<img src="/../README.assets/contact_sequence copy 2.png" alt="contact_sequence copy 2" style="zoom: 67%;" />

<img src="/../README.assets/video_animation copy 2.gif" alt="video_animation copy 2" style="zoom:50%;" />

<img src="/../README.assets/contact_sequence.png" alt="contact_sequence" style="zoom: 67%;" />




## Installation

1. Create a new conda environment:

```bash
conda env create -f env.yaml
```

2. Download ViTPose++ model checkpoint from huggingface

```bash
mkdir easy_ViTPose/checkpoints
wget https://huggingface.co/JunkyByte/easy_ViTPose/resolve/main/torch/ap10k/vitpose-h-ap10k.pth -P easy_ViTPose/checkpoints
```

## Usage

```bash
conda activate gd
python main.py --task_name YOUR_TASK_NAME --video_path YOUR_VIDEO_PATH
```
