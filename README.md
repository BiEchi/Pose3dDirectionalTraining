# Pose3dDirectionalTraining

This is the repo for directionally training the dataset *Human3.6M* on model *VideoPose3D*.

[TOC]

## Code Reproduction

If you want to reproduce the code, please strictly follow the guidance below.

### Prepare Dataset

Extract the archives named `Poses_D3_Positions_S*.tgz` (subjects 1, 5, 6, 7, 8, 9, 11) to a common directory. Your directory tree should look like this:

```
/path/to/dataset/S1/MyPoseFeatures/D3_Positions/Directions 1.cdf
/path/to/dataset/S1/MyPoseFeatures/D3_Positions/Directions.cdf
...
```

### Prepocess The Data

Go to [Code/data/](./Code/data/) and with command 

```shell
# preprocess data in /path/to/dataset
python prepare_data_h36m.py --from-source-cdf /path/to/dataset

# directly retrieve from internet
wget https://dl.fbaipublicfiles.com/video-pose-3d/data_2d_h36m_cpn_ft_h36m_dbb.npz
wget https://dl.fbaipublicfiles.com/video-pose-3d/data_2d_h36m_detectron_ft_h36m.npz
```

### Run The Programme

```shell
cd ..
python3 run.py -e 1 -k cpn_ft_h36m_dbb -arc 3,3,3 -cho Sitting
```

Note that the arguments can be changed. If you want to run a bigger project, simply add up `-e` to 80 and `3,3,3` (27 frames) to `3,3,3,3` (81 frames) or `3,3,3,3,3` (243 frames).



## Comparison Approach

If you want to compare the effect of the original work (*VideoPose3D*) and our work (action-based), you can try running commands below:

### Small-batch, Small-epoch Comparison

```shell
# if you want to estimate our model using the raw data:
for i in Directions Discussion Eating Greeting Phoning Photo Posing Purchases Sitting SittingDown Smoking Waiting WalkDog Walking WalkTogether ;
do
	python run.py -e 15 -k cpn_ft_h36m_dbb -arc 3,3,3 -cho $i ;
done

# if you want to estimate the original model using the raw data:
python run.py -e 1 -k cpn_ft_h36m_dbb -arc 3,3,3 -cho all
```

### Small-batch, Large-epoch Comparison

```shell
# if you want to estimate our model using the raw data:
for i in Directions Discussion Eating Greeting Phoning Photo Posing Purchases Sitting SittingDown Smoking Waiting WalkDog Walking WalkTogether ;
do
	python run.py -e 1200 -k cpn_ft_h36m_dbb -arc 3,3,3 -cho $i ;
done

# if you want to estimate the original model using the raw data:
python run.py -e 80 -k cpn_ft_h36m_dbb -arc 3,3,3 -cho all
```

### Large-batch, Small-epoch Comparison

```shell
# if you want to estimate our model using the raw data:
for i in Directions Discussion Eating Greeting Phoning Photo Posing Purchases Sitting SittingDown Smoking Waiting WalkDog Walking WalkTogether ;
do
	python run.py -e 15 -k cpn_ft_h36m_dbb -arc 3,3,3,3,3 -cho $i ;
done

# if you want to estimate the original model using the raw data:
python run.py -e 1 -k cpn_ft_h36m_dbb -arc 3,3,3,3,3 -cho all
```

### Large-batch, Large-epoch Comparison

```shell
# if you want to estimate our model using the raw data:
for i in Directions Discussion Eating Greeting Phoning Photo Posing Purchases Sitting SittingDown Smoking Waiting WalkDog Walking WalkTogether ;
do
	python run.py -e 1200 -k cpn_ft_h36m_dbb -arc 3,3,3,3,3 -cho $i ;
done

# if you want to estimate the original model using the raw data:
python run.py -e 80 -k cpn_ft_h36m_dbb -arc 3,3,3,3,3 -cho all
```







