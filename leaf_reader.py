import pandas as pd
import numpy as np
import feature_extraction
import os


def one_hot(number, classes=99):
    code = np.zeros(classes, dtype=np.float32)
    code[number] = 1.0
    return code


# returns input data splitted into training and test sets
# labels are one-hot codes of string labels
def readTrainingData(tests_per_species=1, update_descriptors=False, num_descriptors=64):
    data = pd.read_csv("train.csv")
    column = data['species']

    occurrence = dict()
    encoding = dict()

    train_data = []
    test_data = []

    train_labels = []
    test_labels = []

    train_descriptors = []
    test_descriptors = []

    # create map
    for species in column:
        if not (species in occurrence):
            occurrence[species] = 0

    number = 0
    for key in occurrence.keys():
        encoding[key] = one_hot(number)
        number += 1

    if not os.path.exists("descriptors"):
        os.makedirs("descriptors")

    # split to training and test data
    for idx, row in data.iterrows():
        if idx % 10 == 0 and update_descriptors:
            print("Processing " + str(idx) + "...")
        if occurrence[row.values[1]] >= 10 - tests_per_species:
            test_data.append(row.values[2:])
            test_labels.append(encoding[row.values[1]])

            if update_descriptors:
                descriptor = feature_extraction.extract_image_descriptors(
                    'images/' + str(row.values[0]) + ".jpg", num_descriptors)
                np.save("descriptors/" + str(row.values[0]), descriptor)
                test_descriptors.append(descriptor)
            else:
                test_descriptors.append(np.load("descriptors/" + str(row.values[0]) + ".npy"))

        else:
            train_data.append(row.values[2:])
            train_labels.append(encoding[row.values[1]])

            if update_descriptors:
                descriptor = feature_extraction.extract_image_descriptors(
                    'images/' + str(row.values[0]) + ".jpg", num_descriptors)
                np.save("descriptors/" + str(row.values[0]), descriptor)
                train_descriptors.append(descriptor)
            else:
                train_descriptors.append(np.load("descriptors/" + str(row.values[0]) + ".npy"))
        occurrence[row.values[1]] += 1
        # if (count > 5):
        #   break
        # print (pd.get_dummies(row['species']))
        # count+=1

    return (np.asarray(train_data, dtype=np.float32), np.asarray(train_labels),
            np.asarray(test_data, dtype=np.float32), np.asarray(test_labels),
            np.asarray(train_descriptors, dtype=np.float32), np.asarray(test_descriptors))
    # print (np.asarray(test_data).shape)
    # print (data.shape)
    # print (np.asarray(train_data).shape)

    # readTrainingData()
