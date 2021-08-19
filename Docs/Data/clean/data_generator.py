import csv
import random
import pandas as pd

#
# with open('./clean/data.csv', 'a+') as f:
# 	csv_write = csv.writer(f)
# 	for i in range(40):
# 		data_row = ['Ours', 61.9, 69.2, 62, 68.4, 75.7, 88.2, 74.5, 76.9, 81.5, 97.9, 71.1, 80.9, 80.7, 49.9, 59.267]
# 		for j in range(15):
# 			error = random.random() * 1.5
# 			data_row[j+1] += error
# 		csv_write.writerow(data_row)

file = "./data.csv"
df = pd.read_csv(file)
df.T.to_csv(file, header=0, index=0)

