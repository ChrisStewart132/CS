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
    
def main():
    examples = [
        ([2], '-'),
        ([3], '-'),
        ([5], '+'),
        ([8], '+'),
        ([9], '+'),
    ]
    print("classification training data")
    for example in examples:
        print(example)
    distance = euclidean_distance
    combine = majority_element
    for k in range(1, 4, 2):
        print("k =", k)
        print("x", "prediction")
        for x in range(0,10):
            print(x, knn_predict([x], examples, distance, combine, k))
            
    examples = [
        ([1], 5),
        ([2], -1),
        ([5], 1),
        ([7], 4),
        ([9], 8),
    ]
    print("regression training data")
    for example in examples:
        print(example)
    distance = euclidean_distance
    combine = average
    for k in range(1, 4, 2):
        print("k =", k)
        print("x", "prediction")
        for x in range(0,10):
            print("{} {:4.2f}".format(x, knn_predict([x], examples, distance, combine, k)))

    # a training set of y=x**2 from 0-20
    print("predicting values outside of the range of the data-set fails")
    examples = []
    for x in range(0,20,2):
        y = x**2
        input_vector = [x]     
        output_variable = y                                    
        example = (input_vector, output_variable)
        examples.append(example)
    print("y=x**2 regression training data")
    for example in examples:
        print(example)
    distance = euclidean_distance
    combine = average
    for k in range(2,4,5):
        print("k =", k)
        print("x", "prediction")
        for x in range(0,30,5):
            print("{} {:4.2f}".format(x, knn_predict([x], examples, distance, combine, k)))

if __name__ == '__main__':
    import random
    main()










