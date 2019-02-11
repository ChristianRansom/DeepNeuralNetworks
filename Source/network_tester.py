'''
Created on Feb 7, 2019

@author: Christian Ransom
'''

import neuron

def test():
    
    #This is the correct output that the network should eventually learn after enough training
    '''This is a theoretical situation where we can use a single Neuron can learn to 
    recognize which fighters in a game will be strong enough to win a fight. There are
    many different factors which affect the chance a fighter will win. After lots of 
    labeled training, the Network should be able to accurately predict whether or not
    a fighter will win or lose'''
    threshold = 20
    correct_data = []
    correct_data.append(neuron.Input("fast", 5))
    correct_data.append(neuron.Input("strong", 7))
    correct_data.append(neuron.Input("skilled", 4))
    correct_data.append(neuron.Input("tall", 2))
    correct_data.append(neuron.Input("intelligent", 10))

    #I'll manually put in incorrect weights for now. Usually its randomly generated 
    input_data = []
    input_data.append(neuron.Input("fast", 0))
    input_data.append(neuron.Input("strong", 0))
    input_data.append(neuron.Input("skilled", 0))
    input_data.append(neuron.Input("tall", 0))
    input_data.append(neuron.Input("intelligent", 0))

    #print(correct_data)
    #print(inputs)
    
    a_perceptron = neuron.Neuron(input_data, correct_data, threshold)
    a_perceptron.train(100)
    
