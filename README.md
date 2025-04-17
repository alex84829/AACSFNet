# Enhancing Pain Intensity Evaluation via an Attention-Driven Channel-Spatial Fusion Network

## Introduction <a name="Abstract"></a>
The AACSFNet comprises a channel sub-network and a spatial sub-network for feature extraction. The GSFF augments data by integrating facial landmarks and anatomical descriptions into textual prompts, utilizing depth-guided image generation. 

## Overall Pipeline

![architecture](./image/method.png)

The Attention-Aware Channel-Spatial Fusion Network (AACSFNet) comprises three key components:  
1) Channel-subnet (ResNet50-HAM): A modified ResNet50 backbone integrated with a Hybrid Attention Module (HAM) to hierarchically extract and recalibrate channel-wise semantic features.
2) Spatial-subnet (BiScaleCrossViT): A multi-granularity vision transformer with dual-path cross-attention for capturing transient local features and global contextual patterns.
3) The feature fusion block then aligns and integrates their outputs through multi-head attention and residual connections, enabling the model to synthesize discriminative representations for accurate pain intensity regression.  

## Usage

Data

1) data_pre available [here](data_pre.rar: https://pan.baidu.com/s/19OTHCSuixFSAOIqX9uhG6Q?pwd=rdr5    Extracted code: rdr5).
2) Update *path* in the file *data/data_pre* to the path of your dataset.

Installation
1) Clone the repository:<br />
```git clone https://github.com/alex84829/AACSFNet.git``` <br /><br />

2) Install the required dependencies<sup></sup>:<br />
```pip install -r AACSFNet/requirement_AACSFNet.txt```

Run training
Run the following command to train the model.
```bash
python D:\pythonproject\AACSFNet\main_AACSFNet.py
```
## Citation <a name="citation"></a>
"Enhancing Pain Intensity Evaluation via an Attention-Driven Channel-Spatial Fusion Network" has been submitted to the journal The Visual Computer.

@article{AACSFNet,

  title={Enhancing Pain Intensity Evaluation via an Attention-Driven Channel-Spatial Fusion Network},
  
  author={Feng Gao , Linbo Qing , Lindong Li , Ge Yang , Risheng Xu , Li Gao},
  
  journal={The Visual Computer},
  
}

## Acknowledgement

This research was supported by TCI science and technology innovation fund.
