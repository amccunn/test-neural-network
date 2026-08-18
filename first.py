import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

#creating fake rule for network to train on
testSetInputs = np.array([
    [0, 1, 0, 1, 0],
    [1, 0, 1, 1, 0],
    [1, 0, 0, 1, 0],
    [1, 0, 0, 1, 1],
    [0, 0, 0, 0, 0],
    [1, 1, 1, 1, 0],
    [1, 1, 0, 0, 0],
    [0, 0, 1, 1, 1],
    [1, 0, 1, 0, 1],
    [1, 0, 1, 1, 1],
    [0, 0, 1, 1, 1],
    [1, 0, 0, 0, 0],
    [0, 0, 1, 1, 1],
    [1, 0, 1, 0, 1]
])

testSetAnswers = np.array([[0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1]]).T

np.random.seed(69)
weights = np.random.random((5, 1))
learningRate = 0.1

for i in range(1000):
    #forward pass of network to get a node value for each input in the test set
    nodeVals = testSetInputs @ weights
    adjustedOutput = sigmoid(nodeVals)

    #error calculation of the network output vs the expected output
    #error of each test set
    error = testSetAnswers - adjustedOutput

    #backpropagation of the error to adjust the weights
    #the error says how far off the output is from the expected output, and the derivative of the sigmoid function says how much to adjust the weights based on the error
    delta = error * sigmoid_derivative(adjustedOutput)
    #says how much to adjust the specific weight based on the error
    derivative = testSetInputs.T @ delta

    weights += learningRate * derivative

print("Final weights after training:")
print(weights)

unknownInput = np.array([[1, 0, 1, 0, 0]])
unknownNodeVal = sigmoid(unknownInput @ weights)
print("Output for unknown input:")
print(unknownNodeVal)

# #test uniqueness cus im blind
# for i, input in enumerate(testSetInputs):

#     if input in testSetInputs.pop(i):

#         print(f"index {i} is duplicate")

#         break

