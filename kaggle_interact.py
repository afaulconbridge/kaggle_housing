import pandas as pd

# Input data files are available in the "../input/" directory.

# read in everything
data = pd.read_csv('data/train.csv')
data_predict = pd.read_csv('data/test.csv')

y_all = data.SalePrice              
X_all = data.drop(['SalePrice'], axis=1)

# drop manually selected columns
# Id
X_all = X_all.drop(['Id'], axis=1)

# drop columns with too many missing (>25%)
# TODO do this as part of ColumnTransformer
X_all = X_all.dropna(axis=1, thresh=int(0.25*len(X_all)))


numerical_cols = [cname for cname in X_all.columns 
                  if X_all[cname].dtype in ['int64', 'float64']]
categorical_cols = [cname for cname in X_all.columns 
                    if X_all[cname].dtype == "object" 
                    and X_all[cname].nunique() < 8]


X_all[numerical_cols].head()

data_num = data[numerical_cols]
from sklearn.preprocessing import StandardScaler, RobustScaler

import matplotlib
fig1, ax1 = matplotlib.pyplot.subplots()
ax1.boxplot(RobustScaler().fit_transform(data[numerical_cols]), vert=False, 
            labels=data[numerical_cols].columns)
matplotlib.pyplot.show()

# before we can find outliers, need to remove NaN
from sklearn.impute import SimpleImputer
from sklearn.covariance import EllipticEnvelope
from sklearn.svm import OneClassSVM
outliers = OneClassSVM(gamma='scale').fit_predict(
    SimpleImputer(strategy='median').fit_transform(data[numerical_cols]))

# remove outliers
X_all = X_all[outliers == 1]


# look for skew
# need to handle NaN first

from sklearn.impute import SimpleImputer
import scipy.stats
scipy.stats.skewtest(SimpleImputer(strategy='median').fit_transform(data[numerical_cols])).pvalue < 1e-5

data[numerical_cols].columns[scipy.stats.skewtest(SimpleImputer(strategy='median').fit_transform(data[numerical_cols])).pvalue < 1e-5]


# work out if any numeric columns have a majority in a single value
# these could be augmented with a faux categorical

for column in tuple(X_all[numerical_cols].columns):
    values = X_all[column].value_counts()
    max_i = None
    for i in range(len(values)):
        if max_i is None or values[values.axes[0][i]] > values[values.axes[0][max_i]]:
            max_i = i
    if max_i is not None:
        i_count = values[values.axes[0][max_i]]
        i_value = values.axes[0][max_i]
        total_count = sum(values)
        print(column)
        print(i_count)
        print(total_count)
        if i_count > 0.5*total_count:
            # this column should generate an extra column for most commom or not
            # this column has to be numeric
            extra_column = [1 if x == i_value else 0 for x in X_all[column]]
            extra_column_name = "{}_{}".format(column, i_value)
            X_all[extra_column_name] = extra_column

