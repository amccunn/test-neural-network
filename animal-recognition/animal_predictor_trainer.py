import numpy as np
import json

def save_model(network, filename="animal_model_weights.npz"):

    weights_list = []
    biases_list = []
    
    for layer in network.layers:

        # Extract weights and biases from each node in the layer
        layer_weights = [node.weights for node in layer]
        layer_biases = [node.bias for node in layer]
        weights_list.append(np.array(layer_weights))
        biases_list.append(np.array(layer_biases))
        
    # Save arrays into a single file
    np.savez(filename, weights=np.array(weights_list, dtype=object), biases=np.array(biases_list, dtype=object))


