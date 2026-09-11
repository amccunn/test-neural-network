import numpy as np
import json
import glob
from animal_predictor import NeuralNetwork, ANIMALS, edge_finder

if __name__ == "__main__":

    #number of animals to classify
    animals = ANIMALS
    num_classes = len(animals)

    #the target vectors for each animal, where the index of the animal in the ANIMALS list corresponds to the index of the 1 in the target vector
    expected_targets_identifiers = {animal: np.zeros(num_classes) for animal in animals}
    for animal, target in expected_targets_identifiers.items():
        target[animals.index(animal)] = 1.0

    #load the blank model with the correct architecture
    trained_model = NeuralNetwork([[]], model_name="blank_model", learning_rate=0.01)
    trained_model.load_model()

    print("Model loaded successfully. Ready for training.")

    input_data = []
    expected_targets = []

    for filepath in glob.glob("training_data/*.json"):

        #get animal name from filename
        animal_name = filepath.split("/")[-1].split("_")[0]  

        input_data.append(edge_finder(filepath))

        expected_targets.append(expected_targets_identifiers[animal_name])

    trained_model.train(input_data, expected_targets, epochs=10)    