# MSN
Morphing and Sampling Network<br>
[[paper]](https://cseweb.ucsd.edu//~mil070/projects/AAAI2020/paper.pdf)<br>
This code is optimized for users who want to learn only one category competion.

## Environmet
The MDS sampling in parallel is only used in [original environment](https://github.com/Colin97/MSN-Point-Cloud-Completion).<br>
Please check the section of Compile for detail.
### If you use FPS for sampling
- python 3.8.10
- cuda 11.3
- pytorch 1.12.1
- pytorch3d 
- open3d 0.13.0
### If you want to use MDS in parallel
You should use docker
#### make docker images
in MSN dir
```bash
docker build . -t cuda10-0
```
#### make docker container
```bash
docker run --name MSN --gpus all -v [path of MSN in host]:/work -it cuda10-0 /bin/bash
```

## Compile
This code uses extention modules used in [original code](https://github.com/Colin97/MSN-Point-Cloud-Completion).<br>
I can confirm availability of emd and expansion_penalty in python-3.8.10 cuda-11.3 and pytorch-1.12.1. But MDS was not able to use in that environment.

## Dataset
PCN Dataset from [[url]](https://gateway.infinitescript.com/?fileName=ShapeNetCompletion) or [[BaiduYun]](https://pan.baidu.com/share/init?surl=Oj-2F_eHMopLF2CWnd8T3A).(from [[PointTr Github]](https://github.com/yuxumin/PoinTr/blob/master/DATASET.md))
You can use other datasets if it has json file same style with PCN Dataset.

## Usage
training
```python
python train.py
```
evaluation
```python
python eval.py
```
Settings for training and evaluation are in [options.py](https://github.com/GenMNL/MSN/blob/main/options.py).
