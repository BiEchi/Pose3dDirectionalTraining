#!/bin/bash
#BSUB -n 1
#BSUB -R "select[ngpus>0] rusage [ngpus_shared=4]"
#BSUB -e ./Temp/%j.err
#BSUB -o ./Temp/%J.out
#BSUB -q gpuq
#BSUB -J jackposetask

# if you're using the server
cd /gpfsdata/home/zeduoyu/jackfiles/Pose3dDirectionalTraining/VideoPose3D/

# if you're using the repo
# cd /path/to/repo/dir/VideoPose3D

# if you want to estimate the original model using the raw data:
python run.py -e 80 -k cpn_ft_h36m_dbb -arc 3,3,3,3,3 -cho all

# if you want to prepare your dataset:
# cd ./data
# python ./prepare_data_h36m.py --from-source-cdf ../../dataset

# if you want to estimate our model using the raw data:
for i in Directions Discussion Eating Greeting Phoning Photo Posing Purchases Sitting SittingDown Smoking Waiting WalkDog Walking WalkTogether ;
do
    python run.py -e 100 -k cpn_ft_h36m_dbb -arc 3,3,3,3,3 -cho $i ;
done


# if you want to estimate the original model using the best-performance parameter:
# python run.py -k cpn_ft_h36m_dbb -arc 3,3,3,3,3 -c checkpoint --evaluate pretrained_h36m_cpn.bin
