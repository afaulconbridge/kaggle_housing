import pandas as pd
import scipy.stats

from sklearn.pipeline import Pipeline

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, PowerTransformer
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import RFECV
from sklearn.metrics import mean_absolute_error
from sklearn.covariance import EllipticEnvelope
from sklearn.linear_model import LinearRegression, RANSACRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM

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
# reduce prediction data to match train
X_predict = data_predict[X_all.columns]

#determine column types
numerical_cols = [cname for cname in X_all.columns 
                  if X_all[cname].dtype in ['int64', 'float64']]
categorical_cols = [cname for cname in X_all.columns 
                    if X_all[cname].dtype == "object"]
categorical_low_cols = [cname for cname in X_all.columns 
                        if X_all[cname].dtype == "object" 
                        and X_all[cname].nunique() < 6]

skewed = scipy.stats.skewtest(
            SimpleImputer(strategy='median').fit_transform(
                X_all[numerical_cols])
        ).pvalue < 1e-5
numerical_skew_cols = [cname for cname, skew in zip(
                       X_all[numerical_cols].columns, 
                       skewed) if skew]
numerical_nonskew_cols = [cname for cname, skew in zip(
                          X_all[numerical_cols].columns, 
                          skewed) if not skew]


# split into test / train + validation
X_train, X_test, y_train, y_test = train_test_split(
    X_all, y_all, train_size=0.75, test_size=0.25, random_state=0)

# remove outliers in training data (not validation or test)
# this should not be part of the pipeline because, like calculating column
# categories, is a one-off pre-processing step
# was OneClassSVM(gamma='scale')
outliers = IsolationForest(
            n_estimators=512,
            contamination="auto", 
            behaviour="new",
            random_state=0,
            n_jobs=4
            ).fit_predict(
                SimpleImputer(
                    strategy='median'
                ).fit_transform(X_train[numerical_cols])
    )
X_train = X_train[outliers == 1]
y_train = y_train[outliers == 1]
print("Removed {} outliers".format(sum(outliers == -1)))

pipe = Pipeline(
    steps=[
        # TODO remove outliers as preprocess step
        # pre-process to impute and encode
        ("impute", ColumnTransformer(
            transformers=[
                ("num_skew", Pipeline(
                    steps=[
                        ("impute", SimpleImputer(strategy='median')),
                        ("normalize", PowerTransformer(method="yeo-johnson"))
                    ]), numerical_skew_cols),
                ("num_nonskew", Pipeline(
                    steps=[
                        ("impute", SimpleImputer(strategy='mean')),
                        ("normalize", RobustScaler())
                    ]), numerical_skew_cols),
                ("cat", Pipeline(
                    steps=[
                        ("impute", SimpleImputer(strategy='most_frequent')),
                        ("encode", OneHotEncoder(handle_unknown='ignore'))
                    ]), categorical_cols)
            ],
            remainder='drop',
            n_jobs=None
            )),
        # determine optimal features using recursive feature 
        # elimination with cros-validation        
        ("rfe", 
            RFECV(
                estimator=GradientBoostingRegressor(
                    loss='huber',
                    n_estimators=512,  
                    max_depth=32,
                    random_state=0,
                    validation_fraction=0.2,
                    max_leaf_nodes=512,
                    init=LinearRegression()
                ),
                step=0.1,
                cv=4,
                scoring='neg_mean_absolute_error',
                n_jobs=None 
            )),
        # actually do the final model
        ("model", 
            GradientBoostingRegressor(
                loss='huber',
                n_estimators=512,
                max_depth=32,
                random_state=0,
                validation_fraction=0.2,
                max_leaf_nodes=512,
                init=LinearRegression()
            )
        )
    ],
#    memory="/tmp/pipe"
)


param_grid = []
# this generally only works if max_n_estimators > leaf_nodes
for n_estimators, max_leaf_nodes in ((1024, 64), (512, 128), (256, 256)):
    # trade off directly between more trees and bigger trees
    params = {
        'model__n_estimators': [n_estimators],
        'rfe__estimator__n_estimators': [n_estimators],
        'model__max_leaf_nodes': [max_leaf_nodes],
        'rfe__estimator__max_leaf_nodes': [max_leaf_nodes]
    }
    param_grid.append(params)


def mean_std_std(cv_results_):
    best = None
    best_i = None
    for i, result in enumerate(cv_results_):
        score = result['mean_test_score']-(2*result['std_test_score'])
        if best is None or score > best:
            best = score
            best_i = i
    return best_i


# use refit to define a function ranking on mean+std+std 
gsc = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        cv=4, 
        scoring='neg_mean_absolute_error', 
        n_jobs=-1,
        verbose=10,
        iid=False,
        refit=mean_std_std
    )

# execute the grid search - will take a while!
gsc.fit(X_train, y_train)

print("Best parameters found =", gsc.best_params_)

for mean, std, params in zip(
        gsc.cv_results_['mean_test_score'], 
        gsc.cv_results_['std_test_score'], 
        gsc.cv_results_['params']):
    print("  %0.3f (+/-%0.03f) for %r"
          % (-mean, std * 2, params))

    

# print a score based on test data
# this is a score based on unseen data
# pipeline set to the best params and retrained automatically
score = mean_absolute_error(y_test, pipe.predict(X_test))
print("Test mean absolute error =", score)
print("RFE number of features = ", pipe.named_steps['rfe'].n_features_)
print("Number of records =", len(y_all))


# print a score based only on training
# using cross-validation here gives a measure of deviation
scores = cross_val_score(pipe, X_train, y_train, 
                         n_jobs=-1,
                         cv=8,
                         scoring='neg_mean_absolute_error')
# Multiply by -1 since sklearn calculates *negative* MAE
print("Mean of mean absolute error =", -1.0*scores.mean(), 
      "+/-", scores.std()*2.0, "std")


# refit on all data (not just test)
# make more parallel
pipe.set_params(rfe__n_jobs=-1, impute__n_jobs=-1)
# TODO remove outliers from test!
pipe.fit(X_all, y_all)
# predict on results set
predicted = pipe.predict(X_predict)
output = pd.DataFrame({'Id': X_predict.index, 'SalePrice': predicted})
output.to_csv('data/submission.csv', index=False)
