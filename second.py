import numpy as np
from first import sigmoid, sigmoid_derivative

def read_minutes(file_path):

    with open(file_path, 'r') as file:

        #read the data from the file and store it in a list
        data = file.readlines()

    usefulData = [
        (int(line.split(',')[3].split(' ')[1]), int(line.split(',')[10])) for line in data
        if line.split(',')[2] == "Premier League" 
    ]

    newData = [(i, 0) for i in range(1, 39)]

    i = 1
    currentPos = 0

    #fills in the blank gameweeks with 0 minutes played for the player
    while i < len(newData) + 1:

        if currentPos >= len(usefulData):

            break

        elif usefulData[currentPos][0] == i:
        
            newData[i-1] = usefulData[currentPos]
            currentPos += 1
        
        i += 1

    playerMinutes = [i[1] for i in newData]

    return np.array(playerMinutes)

players = ['alison', 'van-dijk', 'codygakkers', 'conor', 'curt', 'dom-szob', 'ekite', 'frimp', 'gravenberch', 'isak', 'jogo', 'kerk', 'keyaysa', 'konte', 'macali', 'mama', 'rio', 'robo', 'salad', 'treyney', 'watend', 'wirtz', 'woodman']


playerMinutesEachGameweek = np.array([read_minutes(f"player-stats/{player}.txt") for player in players])

print("Player minutes for each gameweek:")
print(playerMinutesEachGameweek)