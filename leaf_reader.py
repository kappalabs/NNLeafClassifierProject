import pandas as pd
import numpy as np


def one_hot(number, classes=99):
	code = np.zeros(classes, dtype=np.float32)
	code[number] = 1.0
	return code


# returns input data splitted into training and test sets
# labels are lists of strings, data are numpy arrays
def readTrainingData(tests_per_species=1):

	data = pd.read_csv("train.csv")
	column = data['species']

	occurrence = dict()
	encoding = dict()

	train_data = []
	test_data = []

	train_labels = []
	test_labels = []

	#create map
	for species in column:
		if (not (species in occurrence)):
			occurrence[species]=0

	number = 0
	for key in occurrence.keys():
		encoding[key] = one_hot(number)
		number+=1

	count = 0
	#split to training and test data
	for idx, row in data.iterrows():
		if (occurrence[row.values[1]] >= 10 - tests_per_species):
			test_data.append(row.values[2:])
			test_labels.append(encoding[row.values[1]])

		else:
			train_data.append(row.values[2:])
			train_labels.append(encoding[row.values[1]])
			occurrence[row.values[1]]+=1
		
		#if (count > 5):
		#	break
		#print (pd.get_dummies(row['species']))
		#count+=1


	return (np.asarray(train_data, dtype=np.float32), np.asarray(train_labels), 
			np.asarray(test_data, dtype =np.float32), np.asarray(test_labels))
	#print (np.asarray(test_data).shape)
	#print (data.shape)
	#print (np.asarray(train_data).shape)

#readTrainingData()