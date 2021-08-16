#!/bin/bash
#BSUB -n 1
#BSUB -R "select[ngpus>0] rusage [ngpus_shared=1]"
#BSUB -e ./Temp/%j.err
#BSUB -o ./Temp/%J.out
#BSUB -q gpuq
#BSUB -J jackposetask
cd /gpfsdata/home/zeduoyu/jackfiles/Pose3dDirectionalTraining/VideoPose3D/
python run.py -e 1 -k cpn_ft_h36m_dbb -arc 3,3,3

# if you want to prepare your dataset:
# cd ./data
# python ./prepare_data_h36m.py --from-source-cdf ../../dataset

# if you want to only test whether your code works:
# python run.py -e 1 -k cpn_ft_h36m_dbb -arc 3,3,3

# if you want to train for best performance:
# python run.py -e 80 -k cpn_ft_h36m_dbb -arc 3,3,3,3,3
# if you only want to estimate:

# python run.py -k cpn_ft_h36m_dbb -arc 3,3,3,3,3 -c checkpoint --evaluate pretrained_h36m_cpn.bin
