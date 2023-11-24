# On Fast Simulation of Dynamical System with Neural Vector Enhanced Numerical Solver

![GitHub](https://img.shields.io/github/license/gbup-group/DIANet.svg)


By [Zhongzhan Huang](https://dedekinds.github.io/), [Senwei Liang](https://leungsamwai.github.io), [Hong Zhang](https://scholar.google.com/citations?user=lo_niigAAAAJ&hl=zh-CN), [Haizhao Yang](https://haizhaoyang.github.io/) and [Liang Lin](http://www.linliang.net/).


The official implementation of the technical report paper "On Fast Simulation of Dynamical System with Neural Vector Enhanced Numerical Solver" [[paper](https://arxiv.org/pdf/2208.03680.pdf)]



## Introduction
NeurVec is an open-source and data-driven corrector, which can break through the speed-accuracy trade-off of the large-scale simulations for dynamical systems. NeurVec can be easily plugged into the existing numerical solver, e.g. Euler methond, Runge–Kutta method, etc.

<p align="center">
  <img src="https://github.com/dedekinds/NeurVec/blob/main/image/github.png" width = "830" height = "540">
</p>

## Requirement
* Python 3.7 
* [PyTorch 1.11](http://pytorch.org/)

## Dataset and pre-train model
* Dateset [[Google](https://drive.google.com/drive/folders/1nyiiGsp0QXCsLV1cW8qmH38Bp1VE4mjP?usp=sharing)] [[Baidu](https://pan.baidu.com/s/1-0OXyoJqu_-44xTY47IQJg)] (verification code: g1pd)
* Pre-train model [[Google](https://drive.google.com/drive/folders/1rO59snwIbig2gH-d0Y058-4mUQJ-riUw?usp=sharing)] [[Baidu](https://pan.baidu.com/s/1IYoTwobzFZfZ1Kd3rc76IA)]  (verification code: sixu)


## Citation
If you find this paper helps in your research, please kindly cite 
```
@article{huang2023fast,
  title={On fast simulation of dynamical system with neural vector enhanced numerical solver},
  author={Huang, Zhongzhan and Liang, Senwei and Zhang, Hong and Yang, Haizhao and Lin, Liang},
  journal={Scientific Reports},
  volume={13},
  number={1},
  pages={15254},
  year={2023},
  publisher={Nature Publishing Group UK London}
}
```

## Acknowledgement
We would like to thank zhengdao chen for his pytorch version of [SRNN](https://github.com/zhengdao-chen/SRNN) and Travis for his/her [tutorial](https://travisdoesmath.github.io/pendulum-explainer/).
