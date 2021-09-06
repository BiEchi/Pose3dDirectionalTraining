# LSF Cluster Manipulation

If you’re using an LSF Cluster server, you can use this file for a green-hand guidance.

[TOC]

## Enter GPU Environment

```shell
ssh -XY gpu01 # gpu02~07 are also available
conda activate pose # your name of environment
```

## Check GPU Usage

```shell
nvidia-smi
```

## Run Tasks

```shell
#!/bin/bash
#BSUB -n 1
#BSUB -R "select[ngpus>0] rusage [ngpus_shared=4]"
#BSUB -e %j.err
#BSUB -o %J.out
#BSUB -q gpuq
#BSUB -J jackposetask
cd /gpfsdata/home/zeduoyu/jackfiles/pose/Pose3dDirectionalTraining/Code
python run.py # followed by your arguments for run.py
```

![image-20210816110459000](http://jacklovespictures.oss-cn-beijing.aliyuncs.com/2021-08-16-030459.png)