# AACSFNet
Pain Intensity Evaluation

## Introduction <a name="Abstract"></a>
The AACSFNet comprises a channel sub-network and a spatial sub-network for feature extraction. The GSFF augments data by integrating facial landmarks and anatomical descriptions into textual prompts, utilizing depth-guided image generation. 

## Overall Pipeline

![architecture](./img/method.png)

1) fMRI images parcellated by an atlas to obtain the Functional connectivity matrix for ’N’ ROIs. Rows and columns are rearranged based on community labels of
each ROI to obtain input matrices to local transformer
2) Overview of our local-global transformer architecture: Human brain connectome is a hierarchical structure with ROIs in the same community having greater similarities compared to inter-community similarities. Therefore, we designed a local-global transformer architecture that mimics this hierarchy and efficiently leverages community labels to learn community-specific node embeddings. This approach allows the model to effectively capture both local and global information.
3) Transformer encoder module

## Quick Start Guide <a name="Quick Start Guide"></a>
How to train AACSFNet.

Installation

