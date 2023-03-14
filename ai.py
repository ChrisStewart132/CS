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
        One approach to achieving real-time predictions with k-NN is to precompute the distances between all pairs of training samples and
        store them in a data structure such as a KD tree or ball tree. This allows for fast and efficient retrieval of the k-nearest
        neighbors at prediction time.
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
    
if __name__ == '__main__':
    import random
    main()


