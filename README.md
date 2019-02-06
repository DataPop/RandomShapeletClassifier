# RandomShapeletClassifier

What? 
Classifying timeseries or any other ordered data using shapelets
A shapelet is a subsequence of a series (i.e. a sample from the complete series)

How? 
The trick is to identify shapelets that are representative of a class of timeseries. 
In the old days, people used to
1) calculate each possible shapelet of a timeseries
2) check for each shapelet if it is representative of the timeseries (by checking how often a somewhat similar pattern occurs in the same series) 
3) check if the most representative shapelets of a series can also be found in other timeseries. 
4) If so: these series probably belong to the same class (the degree depends on how often a similar shapelet can be found in a different timeseries or how similar on average a set of shapelets is to shapelets found in other series) 

This is computationally expensive and takes ages. LAME!

In the "randomized" approach, we 
1) sample n shapelets of varying length from random timeseries (ideally n is >> number of series) 
2) we calculate the highest correlation of each shapelet with each timeseries S (we compare each shapelet with each subsample of the same length from all S) 
3) the highest correlation between a shapelet and a timeseries becomes a "feature". So for each series S we eventually have n features. Each feature descibes the correlation between S and each of the n shapelets. 
4) We can now use these features for a standard classification model 

ATTENTION!

Normally the data you put into a classifier would be of shape(N observations x F features) 
Your labels would be of shape (N x 1) 

In this case we classify timeseries (so usually the column-entries). 
So your original data is most likely of shape(N observations x S series). 
The idea is to classify the series and not the single observations. 
So your labels need to be provided in the shape(Sx1).  

So what the code does is to first transform the data from (NxS) to (Sxn). The classification output will be of shape(Sx1) 
N = length of the timeseries
S = number of timeseries in the dataset
n = number of shapelets










