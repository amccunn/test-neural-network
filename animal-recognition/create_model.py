import numpy as np
from animal_predictor import NeuralNetwork, Node, edge_finder, ANIMALS

if __name__ == "__main__":

    filename = "practice.json"

    input_edges = edge_finder(filename)

    flattened_edges = input_edges.flatten()
    input_size = len(flattened_edges)
    hidden_size_1 = 32
    hidden_size_2 = 16

    #number of animals to classify
    animals = ANIMALS
    num_classes = len(animals)

    
    # Hidden Layer 1 (Takes 400 * 400 inputs per node from the flattened edge-detected image) unless the input size is changed
    layer1 = [
        Node(weights=np.random.randn(input_size) * 0.01, bias=0.0, identifier=f"H1_{i}")
        for i in range(hidden_size_1)
    ]

    # Hidden Layer 2 (Takes 32 inputs per node from Layer 1)
    layer2 = [
        Node(weights=np.random.randn(hidden_size_1) * 0.01, bias=0.0, identifier=f"H2_{i}")
        for i in range(hidden_size_2)
    ]

    # Output Layer (Takes 16 inputs per node from Layer 2)
    output_layer = [
        Node(weights=np.random.randn(hidden_size_2) * 0.01, bias=0.0, identifier=animals[i])
        for i in range(num_classes)
    ]

    # Combine into the Deep Neural Network
    deep_animal_net = NeuralNetwork([layer1, layer2, output_layer], model_name="blank_model", learning_rate=0.01)

    deep_animal_net.save_model()