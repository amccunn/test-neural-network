import numpy as np
from first import sigmoid, sigmoid_derivative

def read_data(file_path):

    with open(file_path, 'r') as file:

        #read the data from the file and store it in a list
        data = file.readlines()

    usefulData = [
        line.split(',')[10] for line in data
        if line.split(',')[2] == "Premier League" 
    ]

    return np.array(usefulData)

print(read_data('van-dijk.csv'))
# playersMinutes = np.array([