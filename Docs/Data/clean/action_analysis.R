library(reshape2)
library(ggplot2)

posedata <- data.frame(
  F1UE15VP3D = c( 67.0, 65.8, 68.7, 72.5, 74.2, 87.2, 65.5, 73.1, 85.6, 117, 73.6, 70.8, 81.1, 63.5, 67.3),
  F1UE15Ours = c( 61.9, 69.2, 62.0, 68.4, 75.7, 88.2, 74.5, 76.9, 81.5, 97.9, 71.1, 80.9, 80.7, 49.9, 59.2),
  actions = c('Directions', 'Discussion', 'Eating', 'Greeting', 'Phoning', 'Photo', 'Posing',
              'Purchases', 'Sitting', 'SittingDown', 'Smoking', 'Waiting', 'WalkDog', 'Walking',
              'WalkTogether')
)
posedata

ggplot(data=posedata, mapping=aes(x="model"))+
  geom_bar(stat="count",width=0.5, color='red',fill='steelblue')
