"""
Authors: Mikhail M.Meskhi
Data: 09/03/2021
Title: Assignment 1 - Coding 2
Comments: 


Objectives of the coding assignment:
------------------------------------
(a)
  1. Load the provided dataset.
  2. Implement the distance function.
  3. Implement the neighbors function.
------------------------------------
"""

from math import sqrt
from csv import reader
#the k value considered is 4
#7Th row is used to test the algorithm.
#the following function is uesed to calculate the distance between the value we want to predict(i.e, row 7) to the all the other
#datapoints in the dataset.
def distance(predict_value,row):
	distance = 0.0
	for i in range(len(predict_value) - 1):
		distance += (float(predict_value[i]) - float(row[i])) ** 2
	return sqrt(distance)
	pass
#The function distance() is called in the following function and the values returned from the above function is used to pick
# four datapoints with the least distance from the all.
def get_neighbors(dataset, prediction_row, near_neighbors):
	distances = list()
	for train_row in dataset:
		dist = distance(prediction_row, train_row)
		distances.append((train_row, dist))
	distances.sort(key=lambda tup: tup[1])
	neighbors = list()
	for i in range(near_neighbors):
		neighbors.append(distances[i][0])
	return neighbors
	pass
#the following function does the actual predication of the classification.
def predict(datset, prediction_row, near_neighbors):
	neighbors = get_neighbors(datset, prediction_row, near_neighbors)
	output_values = [row[-1] for row in neighbors]
	#after getting the nearest datapoints the classification which maximum in number is selected and the test row
	#is assigned that value.
	prediction = max(set(output_values), key=output_values.count)
	return prediction
	pass
	
def main():
	#####################
	# Place code here   #
	# Comment the code  #
	#####################
	filename = 'assignment_1_knn_data.csv'
	dataset = list()
	with open(filename, 'r') as file:
		csv_reader = reader(file)
		for row in csv_reader:
			if not row:
				continue
			dataset.append(row)
	dataset.pop(0) # deleting the first row of the dataset which only includes the column names
	prediction = predict(dataset, dataset[7], 4)
	print('The classification it belongs to  %f.',float(prediction))
	pass


if __name__ == '__main__':
	main()

