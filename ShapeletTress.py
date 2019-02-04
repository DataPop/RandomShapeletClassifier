'''
@author: Benny
'''
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sympy.polys.partfrac import apart
from dask.array.random import power
pd.set_option('display.max_columns', 10)
from datetime import datetime
import numpy as np
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix

def sample_column(data):
    return data.columns[np.random.choice(np.array(range(0, len(data.columns))))]
    

def sample_shapelet(timeseries, shapelet_length):
    '''
    draws a random sample from the timeseries
    '''
    i = np.random.choice(np.array(range(0, len(timeseries)-shapelet_length)))
    return np.array(timeseries[i:i+shapelet_length]), timeseries.index[i:i+shapelet_length]


def get_corr(shapelet, subsample):
    '''
    return the correlation between ths shapelet and the subsample that is compared to the shapelet
    '''
    #print np.corrcoef(shapelet, subsample)[0][1]
    return np.corrcoef(shapelet, subsample)[0][1]


def get_max_correlation(timeseries, shapelet):
    '''
    identifies the maximum correlation between a timeseries and a given shapelet
    the maximum correlation is defined as the maximum of all correlations 
    between the shapelet and a subsample of the timeseries with the length of the shapelet
    '''
    as_strided = np.lib.stride_tricks.as_strided
    window = len(shapelet)
    v = as_strided(timeseries, (len(timeseries) - (window - 1), window), (timeseries.values.strides * 2))
    array_list = pd.Series(v.tolist(), index=timeseries.index[:-window+1])
    corr_list = [get_corr(i, shapelet) for i in array_list]
    #max_index = timeseries.index[np.argmax(corr_list):np.argmax(corr_list)+len(shapelet)]
    max_corr = np.max(corr_list)
    #return {'max_index': max_index, 'max_corr': max_corr}
    return max_corr


def evaluate_split_power(CORRELATIONS):
    DF = pd.DataFrame.from_dict(CORRELATIONS, orient = 'Index')
    DF['class'] = DF.index.map(DICT).map(CLASSES)
    dt = tree.DecisionTreeClassifier(max_depth = 1)
    try:
        dt = dt.fit(np.array(DF[DF.columns[0]].reshape(-1, 1), DF['class']))
        split_value = dt.tree_.threshold[0]
        return dt.score(np.array(DF[DF.columns[0]]).reshape(-1, 1), DF['class']), split_value
    except: 
        print 'Missing values! Are filled with nan'
        dt = dt.fit(np.array(DF[DF.columns[0]].fillna(0.0)).reshape(-1, 1), DF['class'])
        split_value = dt.tree_.threshold[0]
        return dt.score(np.array(DF[DF.columns[0]].fillna(0.0)).reshape(-1, 1), DF['class']), split_value


def get_split_power_for_random_shapelet(data, shapelet_length):
    
    column = sample_column(data)
    timeseries = data[column]
    shapelet, shapelet_index = sample_shapelet(timeseries, shapelet_length)
    CORRELATIONS = {}
    for c in data.columns: 
        #if c == column: pass
        #else: 
        CORRELATIONS[c] = get_max_correlation(data[c], shapelet)
    split_power, split_value = evaluate_split_power(CORRELATIONS)
    return column, shapelet, shapelet_index, split_power, split_value, CORRELATIONS

def get_best_shapelet(data, shapelet_length, runs):
    max_split_power = 0.
    for run in range(0, runs):
        print run
        column, shapelet, shapelet_index, split_power, split_value, CORRELATIONS = get_split_power_for_random_shapelet(data, shapelet_length)
        if split_power >= max_split_power: 
            max_split_power = split_power
            best_split_value = split_value
            best_column = column
            best_shapelet = shapelet
            best_shapelet_index = shapelet_index
            correlations = CORRELATIONS
            
    return {'column': best_column, 
            'split_power': max_split_power, 
            'split_value': best_split_value,
            'shapelet': best_shapelet, 
            'shapelet_index': best_shapelet_index,
            'correlations': correlations}
    
    
def plot_discriminant(data, best_column, best_shapelet):
    CORRELATIONS = {}
    for c in data.columns: 
        if c == best_column: pass
        else: 
            CORRELATIONS[c] = get_max_correlation(data[c], best_shapelet)
    
    DF = pd.DataFrame.from_dict(CORRELATIONS, orient = 'Index')
    DF['class'] = DF.index.map(DICT).map(CLASSES)
    dt = tree.DecisionTreeClassifier(max_depth = 1)
    dt = dt.fit(np.array(DF[DF.columns[0]].fillna(0.0)).reshape(-1, 1), DF['class'])
    boundary = dt.predict(np.array(np.arange(0.0, 1.0, 0.01)).reshape(-1, 1))
    DF['predicted'] = dt.predict(np.array(DF[DF.columns[0]].fillna(0.0)).reshape(-1, 1))
    plt.plot(np.arange(0.0, 1.0, 0.01), boundary, label = 'Prediction')
    plt.scatter(DF[DF.columns[0]][DF['class']==0], DF['class'][DF['class']==0], label = 'Actual %s'%REVERSE_CLASSES[0], c = 'k')
    plt.scatter(DF[DF.columns[0]][DF['class']==1], DF['class'][DF['class']==1], label = 'Actual %s'%REVERSE_CLASSES[1], c = 'r')
    plt.xlim(np.min(DF[DF.columns[0]].fillna(0.0))*0.9, np.max(DF[DF.columns[0]].fillna(0.0))*1.1)
    plt.ylim(-0.1, 1.1)
    plt.xlabel('max Correlation of shapelet from %s with other series'%best_column)
    plt.title('Classification Plot')
    plt.legend()
    plt.show()
    

CLASSES = {'vibration':1, 'temperature':0}
REVERSE_CLASSES = {1: 'vibration',0: 'temperature'}

if __name__ == '__main__':
    
    label_df = pd.read_excel('C:/Users/IDM197623/Documents/Data/Fortum/Fortum_MasterData_20181017_HXtags.xlsx')
    label_df = label_df[(label_df['device_id']=='KEE')&(label_df['eng_unit'].isin(['p-p', u'\xb0C']))]
    UNIT_MAP = {'p-p': 'vibration', u'\xb0C': 'temperature'}
    label_df['class'] = label_df['eng_unit'].map(UNIT_MAP)
    DICT = {}
    for i in label_df.index: DICT[label_df.ix[i, 'tag']] = label_df.ix[i, 'class']
    
    '''
    each shapelet is a feature and we want to know the discriminatory power of this feature
    therefore, we ask if splitting the set of timeseries along the max_corr between the shapelet and each timeseries 
    can help us to tell the different classes of timeseries apart
    ideally, we want to build a decision tree in a way that at each node the one shapelet is selected that has the highest discriminatory power
    we want to save this shapelet and the split value (max_corr)
    
    each node is calculated by 
    1) choosing a random timeseries
    2) choosing a random shapelet of length l from this timeseries
    3) calculating the max_corr of this shapelet with all other timeseries in the dataset
    4) testing if this shapelet has discriminatory power
    5) storing the (timeseries_name, shapelet, discriminatory_power) 
    6) repeat n times
    7) choose the shapelet with the highest power to split
    8) repeat 1-7 for the next split within the generated classes
    '''
    
    data = pd.read_csv('C:/Users/IDM197623/Documents/Data/Fortum/2018-07-20/Verdichtet0000000.csv', sep = ';', parse_dates = True)
    data = pd.pivot_table(data[data['device_id']=='KEE'], index = 'ts', columns = 'tag', values = 'avg')
    data.fillna(-1.0, inplace = True)
    #data = pd.read_csv('C:/Users/IDM197623/Documents/Data/Fortum/shapelet_test.csv', sep = ';', parse_dates = True, index_col = 0)
    for c in data.columns: 
        if c in DICT.keys(): 
            if np.std(data[c]) ==0.: 
                del data[c]
                print c, ' is deleted due to no variation'
            else: pass
        else: del data[c]
    
    print data.head()
    
    FOREST_DICT = {}
    for tree_x in range(0, 20):
        shapelet_length = np.random.choice(np.array(range(20, 120)))
        FOREST_DICT['shapelet_feature %s'%tree_x] = get_best_shapelet(data, shapelet_length, runs = 5)
        #plot_discriminant(data, FOREST_DICT['shapelet_feature %s'%tree_x]['column'], FOREST_DICT['shapelet_feature %s'%tree_x]['shapelet'])
    #print FOREST_DICT
    DF = pd.DataFrame.from_dict(FOREST_DICT, orient = 'index')

    LEARN = pd.DataFrame(columns = DF.index, index = DICT.keys())
    LEARN['class'] = LEARN.index.map(DICT).map(CLASSES)
    for c in DF.index: LEARN[c] = LEARN.index.map(FOREST_DICT[c]['correlations'])
    LEARN = LEARN.dropna(axis='index', how='any', thresh = 2)
    LEARN.fillna(-1., inplace = True)
    
    train_cut_off = np.int(round(0.5*len(LEARN), 0))
    test_cut_off = len(LEARN) - train_cut_off
    
    # Set the parameters by cross-validation
    tuned_parameters = [{'n_estimators': [1, 19, 50, 100, 200], 'max_depth': [1, 2, 5, 10],
                     'min_samples_leaf': [1, 2, 5]}]
    model = GridSearchCV(RandomForestClassifier(), tuned_parameters, cv=10, scoring='precision_macro')
    model.fit(LEARN[DF.index].head(train_cut_off), LEARN['class'].head(train_cut_off))
    print(model.best_params_)
    LEARN['prediction'] = model.predict(LEARN[DF.index])
    print LEARN
    print model.score(LEARN[DF.index].tail(test_cut_off), LEARN['class'].tail(test_cut_off))
    print(classification_report(LEARN['class'].tail(test_cut_off), LEARN['prediction'].tail(test_cut_off), target_names = ['temperature', 'vibration']))
    print(confusion_matrix(LEARN['class'].tail(test_cut_off), LEARN['prediction'].tail(test_cut_off)))
    
    
    #rf = RandomForestClassifier()
    cv = cross_val_score(model, LEARN[DF.index], LEARN['class'], cv=25)
    plt.hist(cv)
    plt.show()
    # transform DF to matrix of type: index: series, columns [shapelet_feature | label]
    # apply classification algorithm
    
    # each new time-series wll need to be compared with each shapelet-feature i.o.t. get distance(shapelet / series)
    
