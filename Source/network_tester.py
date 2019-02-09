'''
Created on Feb 7, 2019

@author: Christian Ransom
'''

import perceptron

def test():
    
    threshold = 20
    correct_data = []
    correct_data.append(perceptron.Input("fast", 5))
    correct_data.append(perceptron.Input("strong", 7))
    correct_data.append(perceptron.Input("skilled", 4))
    correct_data.append(perceptron.Input("tall", 2))
    correct_data.append(perceptron.Input("intelligent", 10))

    #I'll manually put in incorrect weights for now. Usually its randomly generated 
    input_data = []
    input_data.append(perceptron.Input("fast", 0))
    input_data.append(perceptron.Input("strong", 0))
    input_data.append(perceptron.Input("skilled", 0))
    input_data.append(perceptron.Input("tall", 0))
    input_data.append(perceptron.Input("intelligent", 0))

    #print(correct_data)
    #print(inputs)
    
    a_perceptron = perceptron.Perceptron(input_data, correct_data, threshold)
    a_perceptron.train(input_data, correct_data, 100)
    
if __name__ == '__main__':
    test()