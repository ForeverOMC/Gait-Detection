# Gait Detection

## Demonstration


| Demonstration                                                      |                          Contact Sequence                          |
| ------------------------------------------------------------------ | :----------------------------------------------------------------: |
| <img src="images/hop.gif" alt="1734874323996" style="zoom:50%;" /> | <img src="images/hop.png" alt="1734874346181" style="zoom:67%;" /> |


| Demonstration                                                        |                           Contact Sequence                           |
| -------------------------------------------------------------------- | :------------------------------------------------------------------: |
| <img src="images/bound.gif" alt="1734874358952" style="zoom:50%;" /> | <img src="images/bound.png" alt="1734874387949" style="zoom:67%;" /> |


| Demonstration                                                        |                           Contact Sequence                           |
| -------------------------------------------------------------------- | :------------------------------------------------------------------: |
| <img src="images/horse.gif" alt="1734874395081" style="zoom:50%;" /> | <img src="images/horse.png" alt="1734874399645" style="zoom:67%;" /> |

## Installation

1. Create a new conda environment:

```bash
conda env create -f env.yaml
```

2. Download ViTPose++ model checkpoint from huggingface

```bash
mkdir easy_ViTPose/checkpoints
cd easy_ViTPose/checkpoints
wget https://huggingface.co/JunkyByte/easy_ViTPose/resolve/main/torch/ap10k/vitpose-h-ap10k.pth
cd ../..
```

## Usage

```bash
conda activate gd
python main.py --task_name YOUR_TASK_NAME --video_path YOUR_VIDEO_PATH
```
