# FD/MMD on audio embedding

## Overview

This repository supports the evaluations of:

- Fréchet Distances (FD)
    - FD_VGG, with [VGGish](https://github.com/tensorflow/models/blob/master/research/audioset/vggish/README.md)
    - FD_PassT, with [PaSST](https://github.com/kkoutini/PaSST)
    - FD_OpenL3, with [OpenL3](https://github.com/torchopenl3/torchopenl3)
    - FD_L-CLAP, with [LAION-CLAP](https://github.com/LAION-AI/CLAP)
    <!-- - FD_EnCodec, with [EnCodec](https://github.com/facebookresearch/encodec) -->

- Maximum Mean Discrepancy (MMD) [Image](https://arxiv.org/pdf/2401.09603), [Music](https://arxiv.org/abs/2503.16669), [Audio](https://arxiv.org/abs/2502.15602)
    - MMD_VGG, with [VGGish](https://github.com/tensorflow/models/blob/master/research/audioset/vggish/README.md)
    - MMD_PassT, with [PaSST](https://github.com/kkoutini/PaSST)
    - MMD_OpenL3, with [OpenL3](https://github.com/torchopenl3/torchopenl3)
    - MMD_L-CLAP, with [LAION-CLAP](https://github.com/LAION-AI/CLAP)
    <!-- - MMD_EnCodec, with [EnCodec](https://github.com/facebookresearch/encodec) -->

- You can refer [FADTK](https://github.com/microsoft/fadtk) for choosing pretrained backbone of audio encoder.

## Installation

### docker 
You can build docker image via dockerfile in `container/dockerfile`.

### Download Pretrained Models

Download [LAION-CLAP checkpoint](https://github.com/LAION-AI/CLAP). Specify path to the checkpoint at --clap_model_path in `run_eval.sh`.

## Runnning evaluation
You can run evaluation by `run_eval.sh`.