#!/bin/bash
#BSUB -n 1
#BSUB -R "select[ngpus>0] rusage [ngpus_shared=1]"
#BSUB -e ./Temp/%j.err
#BSUB -o ./Temp/%J.out
#BSUB -q gpuq
#BSUB -J jackposetask
cd /gpfsdata/home/zeduoyu/jackfiles/Pose3dDirectionalTraining/VideoPose3D/
python run.py -e 1 -k cpn_ft_h36m_dbb -arc 3,3,3,3,3
