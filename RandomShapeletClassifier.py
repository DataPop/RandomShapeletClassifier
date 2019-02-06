from __future__ import division

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


class RandomShapeletForest(BaseEstimator):
    def __init__(self, 
                 number_shapelets = 20, 
                 min_shapelet_length=30, 
                 max_shapelet_length=50):
        """
        
        :param number_shapelets: number of shapelets
        :param min_shapelet_length: minimum shapelet lengths
        :param max_shapelet_length: maximum shapelet lengths
        """
        # Shapelet related
        self.number_shapelets = number_shapelets
        self.min_shapelet_length = min_shapelet_length
        self.max_shapelet_length = max_shapelet_length
        # Training data related
        self.train_data = None
        self.train_labels = None
        self._orig_labels = None
        self.output_size = None
        self.train_size = None
        # validation data
        self.valid_data = None
        self.valid_labels = None
        

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self


    def fit(self, X, y):
        self.number_shapelets = self.number_shapelets
        self.X = X
        self.y = pd.Series(y).values.ravel()
        #self.train_labels = utils.get_one_active_representation(y)
        self.model = self.train_forest()
        return self
    
    
    def predict(self, X):
        self.transformed_data = self.transform(X)
        return self.model.predict(self.transformed_data)

    
    def train_forest(self):
        self.RANDOM_SHAPELETS  = self.get_random_shapelets()
        self.transformed_data = self.transform(self.X)

        tuned_parameters = [{'n_estimators': [1, 19, 50, 100], 'max_depth': [1, 2, 5],
                     'min_samples_leaf': [1, 2, 5]}]
        
        model = GridSearchCV(RandomForestClassifier(), tuned_parameters, cv=2, scoring='precision_macro')
        model.fit(self.transformed_data, self.y)
        return model
    
    def get_random_shapelets(self):
        RANDOM_SHAPELETS = {}
        for i in range(0, self.number_shapelets):
            timeseries = self.sample_column()
            shapelet = self.sample_shapelet(timeseries)
            RANDOM_SHAPELETS['shapelet_%s'%i] = shapelet
        return RANDOM_SHAPELETS
    
    def transform(self, X):
        TRANSFORMED = pd.DataFrame(index = X.keys(), columns = self.RANDOM_SHAPELETS.keys())
        for s in self.RANDOM_SHAPELETS.keys():
            for c in X.columns: 
                TRANSFORMED.loc[c, s] = self.get_max_correlation(X[c], self.RANDOM_SHAPELETS[s])
        
        TRANSFORMED = TRANSFORMED.dropna(axis='index', how='any', thresh = 2)
        TRANSFORMED.fillna(-1., inplace = True)
        return TRANSFORMED  
          
    
    def get_max_correlation(self, timeseries, shapelet):
        '''
        identifies the maximum correlation between a timeseries and a given shapelet
        the maximum correlation is defined as the maximum of all correlations 
        between the shapelet and a subsample of the timeseries with the length of the shapelet
        '''
        as_strided = np.lib.stride_tricks.as_strided
        window = len(shapelet)
        v = as_strided(timeseries, (len(timeseries) - (window - 1), window), (timeseries.values.strides * 2))
        array_list = pd.Series(v.tolist(), index=timeseries.index[:-window+1])
        corr_list = [self.get_corr(i, shapelet) for i in array_list]
        max_corr = np.max(corr_list)
        return max_corr
    
    
    def get_corr(self, shapelet, subsample):
        '''
        return the correlation between ths shapelet and the subsample that is compared to the shapelet
        '''
        #print np.corrcoef(shapelet, subsample)[0][1]
        return np.corrcoef(shapelet, subsample)[0][1]
        
    
    def sample_column(self):
        random_column = self.X.columns[np.random.choice(np.array(range(0, len(self.X.columns))))]
        random_series = self.X[random_column]
        return random_series
        
        
    def sample_shapelet(self, timeseries):
        random_shapelet_length = np.random.choice(range(self.min_shapelet_length, 
                                                             self.max_shapelet_length))
        
        ii = np.random.choice(np.array(range(0, len(timeseries) - random_shapelet_length)))
        random_shapelet = np.array(timeseries[ii:ii + random_shapelet_length])
        return random_shapelet




