import numpy as np
import json
from animal_predictor import NeuralNetwork, ANIMALS

if __name__ == "__main__":

    #number of animals to classify
    animals = ANIMALS
    num_classes = len(animals)

    #the target vectors for each animal, where the index of the animal in the ANIMALS list corresponds to the index of the 1 in the target vector
    expected_targets_identifiers = {animal: np.zeros(num_classes) for animal in animals}
    for animal, target in expected_targets_identifiers.items():
        target[animals.index(animal)] = 1.0

    pre_trained_model = NeuralNetwork([[]], model_name="blank_model", learning_rate=0.01)
    pre_trained_model.load_model()

    print("Model loaded successfully. Ready for training.")

    