import pandas as pd
import numpy as np
from RandomShapelets.RandomShapeletClassifier import RandomShapeletForest

model = RandomShapeletForest(number_shapelets = 10, min_shapelet_length=5, max_shapelet_length=10)
print(model)

data = pd.read_csv('ShapeletForestTest.csv', sep = ';', decimal=b',', index_col = 0)
print(data)
labels = np.array([1., 1., 0., 0.])
print(labels)

m = model.fit(data, labels)

data_2 = pd.DataFrame(index = data.index, columns = ['E', 'F', 'G'])
data_2['E'] = data['D']
data_2['F'] = data['A']
data_2['G'] = data['B']

pred =  m.predict(data_2)
print('should return [0 1 1] ')
print(pred)


