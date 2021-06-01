import numpy as np


def one_hot(data):
    """
    The monk dataset has seven classes on which one-hot encoding is applied to encode categorical information
    #1. class: 0, 1 # These are the labels
    #2. a1: 1, 2, 3
    #3. a2: 1, 2, 3
    #4. a3: 1, 2
    #5. a4: 1, 2, 3
    #6. a5: 1, 2, 3, 4
    #7. a6: 1, 2
    :param data: the seven classes values in list form
    :return: one-hot encoding
    """
    TwoOneHot = {'1': [1, 0], '2': [0, 1]}
    ThreeOneHot = {'1': [1, 0, 0], '2': [0, 1, 0], '3': [0, 0, 1]}
    FourOneHot = {'1': [1, 0, 0, 0], '2': [0, 1, 0, 0], '3': [0, 0, 1, 0], '4': [0, 0, 0, 1]}
    OneHotDict = {'0': ThreeOneHot, '1': ThreeOneHot, '2': TwoOneHot, '3': ThreeOneHot, '4': FourOneHot, '5': TwoOneHot}
    label = data[0]
    features = data[1:]
    oneHotFeatures = []
    for i, feature in enumerate(features):
        oneHotFeatures.extend(OneHotDict[str(i)][feature])
    return oneHotFeatures, int(label)


def extract_data(file_n, file_type):
    """
    :param file_n: number of file
    :param file_type: extension of file
    :return: features and label in ndarray format
    """
    path = 'data/monk/monks-{}.{}'.format(file_n, file_type)
    with open(path) as f:
        features = []
        labels = []
        for line in f.readlines():
            data = line.split(' ')[1:-1]  # Removing irrelevant information
            feature, label = one_hot(data)
            features.append(feature)
            labels.append(label)
        return np.array(features, dtype='float16'), np.array(labels, dtype='float16')