from math import sqrt

"""
k-NN classification/regression: lazy learning, all data in memory, k closest input vectors and their outputs used to predict the unknown output
    Training: Store all the examples (input vector, output variable/vector)
    Prediction: h(x new )
        Let be x1 , . . . , xk the k more similar examples to x new
        h(x new )= combine predictions(x1 , . . . , xk )
            where combine can take the k examples and choose the most common one (majority_element()) or the average vector       
    The parameters of the algorithm are the number k of neighbours and the procedure for combining the predictions of the k examples
    The value of k has to be adjusted (crossvalidation)
        We can overfit (k too low)
        We can underfit (k too high)
    performance:
        One approach to achieving real-time predictions with k-NN is store the training samples in a data structure such as a KD tree or ball tree.
        This allows for fast and efficient retrieval of the k-nearest neighbors at prediction time.
"""
def euclidean_distance(v1, v2):
    return sqrt(sum([(v1[i]-v2[i])**2 for i in range(len(v1))]))
def manhattan_distance(v1, v2):
    return sum([abs(v1[i]-v2[i]) for i in range(len(v1))])
def majority_element(labels):
    """returns the predicted variable most prevalent within the k variables, used for classification problems"""
    d = dict()
    for label in labels:
        if isinstance(label, list):# convert lists/vectors to a tuple to allow for comparisons
            label = tuple(label)
        if label not in d:
            d[label] = 1
        else:
            d[label] += 1
    return sorted(d, key=lambda k:d[k])[-1]
def average(values):
    """returns the average of the k predicted variables/vectors, used for regression problems"""
    if len(values) == 0:
        return 0
    elif isinstance(values[0], (tuple,list)):# output vector
        sum_vector = [0 for variable in values[0]]
        for vector in values:
            for i in range(len(vector)):
                sum_vector[i] += vector[i]
        return [x/len(values[0]) for x in sum_vector]
    else:# output variable
        return sum(values)/len(values)   
def knn_predict(input, examples, distance, combine, k):
    """
    input is a vector of length n
    examples is a list of tuples (x,y) where x is a vector of length n, and y is a vector
    distance/combine are functions (distance: compares input distances, combine: makes an output prediction on the k closest outputs)
    k is the number of saved examples closest to the input used to make an output prediction
    """
    s = sorted(examples, key=lambda x:distance(input,x[0]),reverse=True)
    k_outputs = []
    for i in range(k):
        if len(s) <= 0:
            break# k > len(input/examples) so instead we are using the entire data set
        last_input, last_output = s.pop()# kth closest neighbour input, output from the training set
        k_outputs.append(last_output)# store the kth closest neighbour output to use for prediction
        while len(s) > 0 and distance(s[-1][0], input) == distance(last_input, input):# keep getting other outputs if input distance == kth closest input
            k_outputs.append(s.pop()[1])
    return combine(k_outputs)

"""       
naïve Bayes classification (Gaussian Naive Bayes for regression):
    calculates probability of output "prior" (e.g T=0.43, F=0.47)
    probability for each input variable "likelihood" e.g [((x1=T|output=T),(x1=T|output=F))...xn]
    an input vector has its probability for the output calculated based on the training data/calculated probabilities
        the probability if > some_threshold (e.g 0.5) can return T else F


Representation of naïve Bayes models

Naïve Bayes models can be represented with belief networks. However, since they all have a very simple topology
(a directed tree of depth one where the root is the class variable and the leaves are the input features),
we can use a more compact representation that is only concerned with the values of CPTs.

We assume that all the variables in a naïve Bayes network are binary. For a network with n binary input features X[1] to X[n],
we represent the conditional probability tables (CPTs) that are required in the network, with the following two objects:

    prior: a real number representing p(Class=true). The probability p(Class=false) can be obtained by 1 - prior.
    likelihood: a tuple of length n where each element is a pair of real numbers such that likelihood[i][False] is p(X[i]=true|C=false)
    and likelihood[i][True] is p(X[i]=true|C=true ). That is, likelihood contains the 2*n CPTs that are required at leaf nodes.

Note: in general, indexing sequences with booleans is not ideal, however, here we are using False (for 0) and True (for 1) so that the
appearance of the code is closer to the corresponding mathematical notation.
"""
def posterior(prior, likelihood, observation):
    t = prior
    f = 1-prior
    for i, x in enumerate(likelihood):
        if observation[i]:
            pt = (x[observation[i]])
            pf = (x[not observation[i]])
        else:
            pt = 1-(x[not observation[i]])
            pf = 1-(x[observation[i]])
        t *= pt
        f *= pf
    t /= (t+f)
    return t

import csv
"""
given a csv / excel spreadsheet with the top row being x variables (e.g. x1,x2...x12) with an extra variable for the output classification
    example csv (x1="money" present, x2="dear" present... classification = spam(T or F)
        x1 x2 x3 x4 classification
        1  0  1  1  1
        0  1  1  1  0
        1  0  0  1  1
        0  0  0  0  0
"""
def learn_prior(file_name, pseudo_count=0):
    """
    calculates the datasets classification probability
    """
    with open(file_name) as in_file:
        training_examples = [tuple(row) for row in csv.reader(in_file)]
    total = 0#count of true class variable (last one in csv file) + p_c / false class var + p_c*len(domain)
    for i,line in enumerate(training_examples):
        temp = line[-1]
        #print(temp)
        if i > 0:
            total += int(temp)       
    return (total + pseudo_count) / (len(training_examples)-1 + pseudo_count*2)
def learn_likelihood(file_name, pseudo_count=0):
    """
    calculates the dataset probability if the classification is True and False for each variable
        e.g [((x1=T|output=T),(x1=T|output=F))...xn]
    """
    with open(file_name) as in_file:
        training_examples = [tuple(row) for row in csv.reader(in_file)]
    class_var_true_count = 0
    output = [[0,0] for x in range(len(training_examples[0])-1)]#[ [0,0],[0,0],[0,0] ] list of likelihoods
    
    for line in training_examples[1:]:
        b = int(line[-1])# class variable F/T - 0/1
        if b:
            class_var_true_count += 1
            
        for i, variable in enumerate(line[:-1]):        
            if b:
                output[i][1] += int(variable)              
            else:
                output[i][0] += int(variable)
    for i in range(len(output)):
        output[i][1] += pseudo_count
        output[i][1] /= (class_var_true_count + pseudo_count*2)
        
        output[i][0] += pseudo_count
        output[i][0] /= ((len(training_examples)-1-class_var_true_count) + pseudo_count*2)
    return output
def nb_classify(prior, likelihood, input_vector):
    """
    based on the learned prior and likelihood probabilities from the data-set
    estimates a probability that the input_vector is spam or not spam
    """
    x = posterior(prior, likelihood, input_vector)
    l = "spam" if x > 0.5 else "not spam"# choose to predict as spam or not spam
    x = x if x > 0.5 else 1-x# probability of spam or not spam
    return l,x


"""
artificial neurual network classification/regression:
    perceptron (weight=[bias,w1,w2,...,wn], input=[1,x1,x2,...,xn]):
        return T if sum([w[i]*in[i] for i in range(len(weight))]) > 0 else F
    a classification perceptron emits T=1/F=0 depending on its bias and how it weighs each input variable x
    a regression perceptron emits its numerical weighting

    perceptron structure: n input->layers of perceptrons->output

    perceptrons are trained on the dataset, updating weight to get closer to expected output

    unknown input vectors parsed through network predicting output
"""
def construct_perceptron(weights, bias=0.5, type=0):
    """Returns a perceptron function using the given paramers."""
    if type == 0:
        def perceptron(input):# binary classification perceptron, outsputs {0,1}
            output = sum([weights[i]*input[i] for i in range(len(weights))])
            # Note: we are masking the built-in input function but that is
            # fine since this only happens in the scope of this function and the
            # built-in input is not needed here.
            return int(output + bias >= 0)# activation function
    else:
        def perceptron(input):# multi-classification perceptron, outputs {probability_n for n in input} using softmax activation function
            #sigmoid function takes input x, outputs y from 0 < y < 1 
                # f(x) = 1/(1+e**-x), where e = 2.71828, x <= -6 maps to 0, x >= 6 maps to 1...(f(0) = 0.5)
            output = sum([weights[i]*input[i] for i in range(len(weights))]) + bias
            output = min(output, 6) if output > 6 else output
            output = max(output, -6) if output < -6 else output
            return 1/(1+(2.71828**-output))# activation function 
    return perceptron # this line is fine

def perceptron_accuracy(classifier, inputs, expected_outputs):
    """classifier=perceptron(input) function, inputs = list of input vectors, expected_outputs = list of corresponding output variables/vectors"""
    outputs = [1 if x[0] == '+' else 0 for x in expected_outputs]# convert expected_outs to binary {0,1} domain to compare with binary perceptron
    comparisons = []
    for i in range(len(inputs)):
        prediction = '+' if classifier(inputs[i]) == 1 else '-'
        print(inputs[i], prediction, outputs[i] == classifier(inputs[i]))
        comparisons.append(outputs[i] == classifier(inputs[i]))
    return (sum(comparisons) / len(comparisons))*100

def learn_perceptron_parameters(weights, bias, training_examples, learning_rate=0.1, max_epochs=1):
    for epoch in range(max_epochs):# for each cycles / repition of learning
        for example in training_examples:# for each test case
            perceptron = construct_perceptron(weights, bias)
            x = example[0]# real input vector
            t = 1 if example[1][0] == '+' else 0# real output 
            y = perceptron(x)# predicted output
            for i in range(len(x)):# for each input variable adjust its associated weight       
                weights[i] += learning_rate*x[i]*(t-y)# weight vector corrected / learning
            bias += learning_rate*(t-y)# bias corrected / learning
    return weights, bias# return the corrected weights vector and bias to construct a new perceptron for making predictions



    
def main():
    # training set (classify positive and negative integers between -2**8<->2**8-1
    sorted_signed_integers = sorted([random.randint(-2**8,(2**8)-1) for i in range((((2**8)-1)+2**8) // 16)])
    signed_integer_data_set = []# tuple (input_vector, output_vector)
    for n in sorted_signed_integers:
        input, output = [n], ['+'] if n >= 0 else ['-']
        signed_integer_data_set.append((input, output))

    training_set, testing_set = [], []# training set is 2/3 of the entire data set, testing set is 1/3
    for i, example in enumerate(signed_integer_data_set):
        if i % 3 == 2:
            testing_set.append(example)
        else:
            training_set.append(example)            
    print("signed integer classification data_set")
    print(signed_integer_data_set)  
    print("\nsigned integer classification training_set")
    print(training_set)
    print("\nsigned integer classification testing_set")
    print(testing_set,'\n')

    # confirm that the machine learning algorithms accurately predict the testing set after being trained
    
    # k-NN (k Nearest Neighbours)
    k = 3
    print("k-NN (k Nearest Neighbours): k =", k)
    print("x", "prediction testing")
    accuracy = 0
    for input, output in testing_set:
        predicted_output = knn_predict(input, training_set, euclidean_distance, majority_element, k)
        print(input, predicted_output, predicted_output == tuple(output))
        accuracy += 1 if predicted_output == tuple(output) else 0
    accuracy /= len(testing_set)
    print("accuracy:", accuracy*100,"%")



    # ANN (Artificial Neural Network)
    print("\nANN (Artificial Neural Network)")
    
    # initialize weights vector and bias value to any random number(s)
    weights, bias = [random.randint(-10,10) for x in range(len(testing_set[0][0]))], random.randint(-10,10)
    print("initial", "weights:", weights, "bias:", bias)
    
    # train the perceptron on the training_set and return the calibrates weights vector and bias value
    weights, bias = learn_perceptron_parameters(weights, bias, training_set, learning_rate=0.5, max_epochs=500)
    print("trained", "weights:", weights, "bias:", bias)
    
    # construct a perceptron with the trained weights vector and bias value
    perceptron = construct_perceptron(weights, bias)

    # extract the testing_inputs and testing_outputs from the testing set
    testing_inputs, testing_outputs = [x[0] for x in testing_set], [x[1] for x in testing_set]

    # test how accurately the single perceptron is able to predict the sign of an integer
    accuracy = perceptron_accuracy(perceptron, testing_inputs, testing_outputs)
    print("accuracy:", accuracy, "%")













    # multi dimensional input with single output variable
    # training set (classify positive and negative integers between -2**8<->2**8-1
    signed_integers = [[random.randint(-2**8,(2**8)-1) for j in range(3)] for i in range((((2**8)-1)+2**8) // 16)]
    sorted_signed_integers = sorted(signed_integers, key=lambda x:(x[0], x[1], x[2]))
    signed_integer_data_set = []# tuple (input_vector, output_vector)
    for n in sorted_signed_integers:
        input = n
        output = '+-'# +, -, or +-
        if all([x>=0 for x in input]):
            output = '+'
        elif all([x<0 for x in input]):
            output = '-'
        signed_integer_data_set.append((input, [output]))

    training_set, testing_set = [], []# training set is 2/3 of the entire data set, testing set is 1/3
    for i, example in enumerate(signed_integer_data_set):
        if i % 3 == 1:
            testing_set.append(example)
        else:
            training_set.append(example)
    print("\n\n\nsigned integer classification data_set")
    print(signed_integer_data_set)  
    print("\nsigned integer classification training_set")
    print(training_set)
    print("\nsigned integer classification testing_set")
    print(testing_set,'\n')

    # confirm that the machine learning algorithms accurately predict the testing set after being trained
    k = 3
    print("k-NN: k =", k)
    print("x", "prediction testing")
    accuracy = 0
    for input, output in testing_set:
        predicted_output = knn_predict(input, training_set, euclidean_distance, majority_element, k)
        print(input, predicted_output, predicted_output == tuple(output))
        accuracy += 1 if predicted_output == tuple(output) else 0
    accuracy /= len(testing_set)
    print("accuracy:", accuracy*100,"%")



    # ANN (Artificial Neural Network) MLP (multi-layer perceptrons)
    print("\nANN (Artificial Neural Network)  MLP (multi-layer perceptrons)")

    # 3 input variables, 1 output variable, therefore 3 inputs -> 3 hidden perceptron -> 1 output perceptron

    # 3 hidden perceptrons, each take in the original input vector and output 0<->1
    p1 = construct_perceptron([6,0,0], 0, 1)# outputs x1 mapped to 0<->1, < 0.5 implies x1 is negative, >= 0.5 implies x1 positive
    p2 = construct_perceptron([0,6,0], 0, 1)# p2 and p3 follow the same logic but for x2, and x3 respectively
    p3 = construct_perceptron([0,0,6], 0, 1)

    # output perceptron has three inputs each in the range from 0<->1, it outputs whether the original input is all positve,all negative,or neither(+-)
    output_perceptron = construct_perceptron([1,1,1], 0, 1)# all inputs are in the range 0<->1
        # therefore max input = sigmoid(3) ~= 0.95, min output = 0.5
        # for simplicity, <0.55 == -, > 0.9==+, else +-

    hidden_perceptrons = [p1,p2,p3]
    accuracy = 0
    for input, output in testing_set:      
        hidden_outputs = []
        for i in range(len(hidden_perceptrons)):# for each hidden perceptron
            hidden_outputs.append(hidden_perceptrons[i](input))
        predicted_output = output_perceptron(hidden_outputs)
        symbol = '+-'
        if predicted_output < 0.55:
            symbol = '-'
        elif predicted_output > 0.9:
            symbol = '+'
        accuracy += 1 if symbol == output[0] else 0
        print(input, symbol, symbol==output[0]) 
    print("accuracy:", (accuracy/len(testing_set))*100, "%")












    
    # multi dimensional input and output vectors
    # training set (classify positive and negative integers between -2**8<->2**8-1
    signed_integers = [[random.randint(-2**8,(2**8)-1) for j in range(3)] for i in range((((2**8)-1)+2**8) // 16)]
    sorted_signed_integers = sorted(signed_integers, key=lambda x:(x[0], x[1], x[2]))
    signed_integer_data_set = []# tuple (input_vector, output_vector)
    for n in sorted_signed_integers:
        input = n
        output = ['+' if x >= 0 else '-' for x in n]
        signed_integer_data_set.append((input, output))

    training_set, testing_set = [], []# training set is 2/3 of the entire data set, testing set is 1/3
    for i, example in enumerate(signed_integer_data_set):
        if i % 3 == 1:
            testing_set.append(example)
        else:
            training_set.append(example)
    print("\n\n\nsigned integer classification data_set")
    print(signed_integer_data_set)  
    print("\nsigned integer classification training_set")
    print(training_set)
    print("\nsigned integer classification testing_set")
    print(testing_set,'\n')

    # confirm that the machine learning algorithms accurately predict the testing set after being trained
    k = 3
    print("k-NN: k =", k)
    print("x", "prediction testing")
    accuracy = 0
    for input, output in testing_set:
        predicted_output = knn_predict(input, training_set, euclidean_distance, majority_element, k)
        print(input, predicted_output, predicted_output == tuple(output))
        accuracy += 1 if predicted_output == tuple(output) else 0
    accuracy /= len(testing_set)
    print("accuracy:", accuracy*100,"%")


    # ANN (Artificial Neural Network) MLP (multi-layer perceptrons)
    print("\nANN (Artificial Neural Network)  MLP (multi-layer perceptrons)")

    # 3 input variables, 3 output variable, therefore 3 inputs -> 3 hidden perceptron -> 3 output perceptrons

    # 3 hidden perceptrons, each take in the original input vector and output 0<->1
    p1 = construct_perceptron([6,0,0], 0, 1)# outputs x1 mapped to 0<->1, < 0.5 implies x1 is negative, >= 0.5 implies x1 positive
    p2 = construct_perceptron([0,6,0], 0, 1)# p2 and p3 follow the same logic but for x2, and x3 respectively
    p3 = construct_perceptron([0,0,6], 0, 1)
    # 3 output perceptrons
    o1 = construct_perceptron([6,0,0], -0.5, 1)# output perceptrons recieve 0<->1 so bias of -0.5 to make the range -0.5<->0.5
    o2 = construct_perceptron([0,6,0], -0.5, 1)
    o3 = construct_perceptron([0,0,6], -0.5, 1)

    hidden_perceptrons = [p1,p2,p3]
    output_perceptrons = [o1,o2,o3]
    accuracy = 0
    for input, output in testing_set:      
        hidden_outputs = []
        for i in range(len(hidden_perceptrons)):# for each hidden perceptron
            hidden_outputs.append(hidden_perceptrons[i](input))
            
        predicted_outputs = []
        for i in range(len(output_perceptrons)):
            predicted_output = output_perceptrons[i](hidden_outputs)
            symbol = '+' if predicted_output >= 0.5 else '-'
            predicted_outputs.append(symbol)

        accuracy += 1 if predicted_outputs == output else 0
        print(input, predicted_outputs, predicted_outputs==output) 
    print("accuracy:", (accuracy/len(testing_set))*100, "%")





    print("\n\n\nregression training data")
    # a training set of y=x**2 from 0-20  
    examples = []
    for x in range(0,20,2):
        y = x**2
        input_vector = [x]     
        output_variable = y                                    
        example = (input_vector, output_variable)
        examples.append(example)
    print("y=x**2 regression training data")
    print(examples)
    k=3
    print("k =", k)
    print("x", "prediction", "accuracy(distance)")
    for x in range(0,30,5):
        predicted_output = knn_predict([x], examples, euclidean_distance, average, k)
        print("{} {:4.2f} {:.2f}".format(x, predicted_output, abs(x**2-predicted_output)))

            
if __name__ == '__main__':
    import random
    main()










