"""
chatgpt:
There are several common non-linear function extrapolation methods, including:

    Runge-Kutta methods: These are numerical methods used to solve ordinary differential equations, including non-linear functions.

    Adams-Bashforth methods: These are numerical methods used to solve differential equations, including non-linear functions.

    Finite difference methods: These methods approximate derivatives using finite differences and can be used to extrapolate non-linear functions.

    Non-linear regression: This involves fitting a non-linear function to a set of data points to extrapolate the function beyond the given data points.

    Neural networks: These are machine learning models that can be trained on a set of data points to extrapolate non-linear functions.
    
Sure, here are some more examples of common non-linear function extrapolation methods:
    Simpson's rule
    Monte Carlo simulation
    Taylor series expansion
    Gradient descent
    Markov chain Monte Carlo (MCMC) methods
    Nonlinear least squares fitting
    Genetic algorithms
    Simulated annealing
    Artificial neural networks
    Support vector machines
    Decision trees
    Random forests
    Gradient boosting
    Deep learning models (e.g. convolutional neural networks, recurrent neural networks)

Here are some common methods to extrapolate on sequences:
    Linear extrapolation
    Polynomial extrapolation (including quadratic, cubic, and higher-order polynomials)
    Exponential extrapolation
    Logarithmic extrapolation
    Power-law extrapolation
    Fourier extrapolation
    Moving average extrapolation
    Autoregressive (AR) extrapolation
    Moving average autoregressive (ARMA) extrapolation
    Kalman filtering
    Neural network extrapolation
    Support vector machine (SVM) extrapolation
    Gaussian process extrapolation
    Hidden Markov model (HMM) extrapolation
    Long short-term memory (LSTM) extrapolation.

There are several methods to extrapolate on a time series, including:

    Simple linear regression: a statistical method that assumes a linear relationship between the dependent variable (y) and one independent variable (x), and uses the least squares method to estimate the intercept and slope of the regression line.

    Exponential smoothing: a technique that uses a weighted average of past observations to forecast future values. The weights decrease exponentially as the observations get older, giving more importance to recent data.

    ARIMA (AutoRegressive Integrated Moving Average) models: a class of statistical models that can be used to describe and predict time series data. ARIMA models use the past values of the time series and the errors from past predictions to estimate future values.

    Seasonal decomposition: a method that separates a time series into its trend, seasonal, and residual components, and models each component separately.

    Neural networks: a type of machine learning algorithm that can learn complex patterns in data, and can be trained to predict future values of a time series.

    Fourier analysis: a mathematical method that decomposes a time series into its frequency components, and can be used to identify cyclical patterns in the data.

    Holt-Winters method: a time series forecasting method that uses exponential smoothing to model the trend and seasonal components of a time series.
"""



def y(t):
    #return 64.23*t**4 - 31.87*t**3 + 4.52*t**2 - 9.81*t - 75.1
    return 3*t**5 - 4.23*t**4 - 9.87*t**3 - 4.52*t**2 - 9.81*t - 75.1
    #return 12.1**t# can not predict exponential functions 

def diff_list(l):
    return [l[i]-l[i-1] for i in range(1,len(l))]

def predict_next_point_from_points(points):
    """
    given a series of points with a uniform difference in time between them (delta_t), predicts the future point delta_t seconds in the future
    the more points supplied, the greater the degree of derivitives used

    usage:
        wait until n points are accumulated (e.g. n=2 to accout for velocity, 3 for acceleration...) with each point having a uniform
            difference in time between measurements
        iterate on the difference in time predicting the next point and recursively use the last n points including the predicted points
            until the predicted point is numerically estimated at the correct future time.
        all known and predicted points could be cached to account for on the fly time changes (refer to cache for earlier forecasts)
            or predict furthur into the future to anticipate potential time changes.

    could possibly compare predictions to real outcomes once the time has elapsed, and adjusting n or some weight function (lazy machine learning)
        e.g. test 1/40s into the future, once for each value of n, compare the real to predicted and choose the best n
    """
    #min_val = 0.000000000001# smallest value to prevent 
    diff_lists = [points]
    for i in range(1, len(points)):
        diff_lists.append(diff_list(diff_lists[-1]))
    p = 0# predicted point
    for i in range(len(diff_lists)-1,-1,-1):
        #if abs(diff_lists[i][-1]) > min_val:
        p += diff_lists[i][-1]# if delta_t between the given points were not uniform, time scaling would be needed here

    return p

# psuedo function
def n_accuracy(n=2, n_points=5):
    # accumulate points every delta_t seconds up to len(n) for prediction
    # continue accumulating delta_t seconds to find next point(s) for comparison
    while len(points) < n + n_points:# wait for the next n_points to test against
        points.append(0)# append the recorded positions/outputs
    # using the first n points recorded predict the next n_points

    # compare the predicted n_points to the recorded n_points
    
    return 0# return the last points error (or the largest error from all points predicted, or the sum of all errors (mean squared error or abs error))
        

delta_t = 1/40
i = 10
p0=[]
while len(p0) < 10:
    p0.append(y(i))
    i += delta_t



p1 = diff_list(p0[-5:len(p0)])  # 1st derivitive, velocity... only require 5 positions, len(p1)=4..., p2=3, p3=2, p4=1
p2 = diff_list(p1)              # 2nd derivitive, acceleration
p3 = diff_list(p2)              # 3rd derivitive, jerk
p4 = diff_list(p3)              # 4th derivitive, snap

print("original p0 data")
print(p0)

print("\np1,p2,p3,p4 difference vectors")
print(p1)
print(p2)
print(p3)
print(p4)
print("\n")

e_p0 = [x for x in p0]

forecasting_time = 1 # comparing predictions 5s into the future with real values

t = i
while t < i+forecasting_time:
    if False:# manual
        s,j,a,v,p = 0,0,0,0,0

        e_p1 = diff_list(e_p0[-5:len(e_p0)])# 1st derivitive, velocity
        e_p2 = diff_list(e_p1)              # 2nd derivitive, acceleration
        e_p3 = diff_list(e_p2)              # 3rd derivitive, jerk
        e_p4 = diff_list(e_p3)              # 4th derivitive, snap
        
        s = e_p4[-1]
        j = e_p3[-1] + s
        a = e_p2[-1] + j
        v = e_p1[-1] + a
        p = e_p0[-1] + v
        e_p0.append(p)
    else:# using predict_next_point_from_points()
        e_p0.append(predict_next_point_from_points(e_p0[-6:len(e_p0)]))
    t += delta_t

start = i
while i < start+forecasting_time:
    p0.append(y(i))
    i += delta_t
    
error = [p0[i] - e_p0[i] for i in range(len(p0))]
for i in range(len(p0)):
    print(f"{p0[i]:.3f}, predicted: {e_p0[i]:.3f}, error: {error[i]:.12f}, time:{i*delta_t}")
