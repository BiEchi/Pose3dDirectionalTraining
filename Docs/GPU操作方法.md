直接进入GPU环境：ssh -XY gpu0X, X=(1~7)，然后激活环境进入文件夹
使用脚本命令（自动判断是否）进入GPU环境：

例程如下
#!/bin/bash
#BSUB -n 1
#BSUB -R "select[ngpus>0] rusage [ngpus_shared=2]"
#BSUB -e %j.err
#BSUB -o %J.out
#BSUB -q gpuq
#BSUB -J jackposetask
cd /gpfsdata/home/zeduoyu/jackfiles/pose/VideoPose3D/
python run.py

![IMG_B08438C31EB2-1](http://jacklovespictures.oss-cn-beijing.aliyuncs.com/2021-08-15-112857.jpg)