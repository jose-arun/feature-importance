import pandas
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

import eli5
from eli5.sklearn import PermutationImportance


import xgboost as xgb
import matplotlib.pyplot as plt
from textwrap import wrap

# from skater.core.explanations import Interpretation
# from skater.model import InMemoryModel

# Choose dataset
# df = pandas.read_csv('dataset/20191210duo_data_by_word.csv')
df = pandas.read_csv('dataset/20191120duo_data_assembled_logged.csv')

print("Number of rows of the complete dataset: " + str(df.shape[0]))

# Impute missing values for Human Concreteness
# Returns the list of empty values per column
print("Number of null values before imputation in Human Concreteness: " + str(df['human_conc'].isnull().sum()))
imp_mean = SimpleImputer( strategy='mean')
df['human_conc'] = imp_mean.fit_transform(df[['human_conc']]).ravel()
print("Number of null values after imputation in Human Concreteness: " + str(df['human_conc'].isnull().sum()))

# Select relevant columns
total_columns = ['wordexperience_log', 'userexperience_log', 'human_conc', 'ld',
                    'local_alignment_avg', 'semantic_density_l1', 'l1_freq_wfzipf',
                    'l1_nrsynsets_wn', 'propcorr_p'] #TODO:  'POS', 'lang_pair'
df = df[total_columns]

# One-hot encoding for lang_pair


# Drop the missing values
df = df.dropna()
print("Number of rows of the dataset after dropping NaN rows: " + str(df.shape[0]))

# Train-Test Split
train, test = train_test_split(df, test_size=0.01);

# Extract the predictors and the label to be predicted, split into train and test
X_train = train.loc[:, df.columns != 'propcorr_p']
y_train = train.loc[:, df.columns == 'propcorr_p']

X_test = test.loc[:, df.columns != 'propcorr_p']
y_test = test.loc[:, df.columns == 'propcorr_p']

print("Shape of X_train: " + str(X_train.shape))
print("Shape of y_train: " + str(y_train.shape))
print("Shape of X_test: " + str(X_test.shape))
print("Shape of y_test: " + str(y_test.shape))

# Standardize the data attributes
X_train = preprocessing.scale(X_train)
X_test = preprocessing.scale(X_test)


# Print the predictors
predictor_columns = total_columns[:-1] # The last column is the value to be predicted
print("The predictors used in the model are: ")
print(predictor_columns)

fig = plt.figure()
ax = fig.add_subplot(111)  # The big subplot
ax1 = fig.add_subplot(421)
ax2 = fig.add_subplot(423)
ax3 = fig.add_subplot(424)
ax4 = fig.add_subplot(425)
ax5 = fig.add_subplot(426)
ax6 = fig.add_subplot(427)
ax7 = fig.add_subplot(428)
# fig.delaxes(ax[0,1]) # Deleting the unused slot
# Plotting function
def plot_feature_importance_model(feature_importance, modelName, axis):
    global predictor_columns
    print("Predictor Columns : " + str(predictor_columns))
    print(modelName + " feature importances: " + str(feature_importance))
    # Shrinking the label sizes so that they don't overlap
    plt.rcParams.update({'font.size': 1})
    cropped_predictor_columns = ['\n'.join(wrap(l, 11)) for l in predictor_columns]

    # Turn off axis lines and ticks of the big subplot
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)

    axis.bar(cropped_predictor_columns, feature_importance)
    axis.set_title(modelName, position=(0.8,0.75))
    for item in ([axis.title, axis.xaxis.label, axis.yaxis.label] +
                 axis.get_xticklabels() + axis.get_yticklabels()):
        item.set_fontsize(6)

    # Set common labels
    ax.set_xlabel('Predictor', fontsize=12)
    ax.set_ylabel('Feature Importance measure', fontsize=12)
    ax.set_title('Plotting the feature importance of predictors \n in L2 word learning (complete dataset)', fontsize=15)

    #
    # ax[subplot_x, subplot_y].bar(cropped_predictor_columns, feature_importance)
    #
    # # plt.bar(predictor_columns, feature_importance)
    # plt.xlabel('Predictor', fontsize=10)
    # plt.ylabel('Relative feature importance', fontsize=10)
    # # plt.title('Plotting the feature importance of predictors \n in L2 word learning using ' + modelName, fontsize=12)


# Predict with LR
print("******************** LINEAR REGRESSION ********************")
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
print("R squared for the Linear Regression model: " + str(r2_score(y_test, y_pred)))
print("MSE for the Linear Regression model: " + str(mean_squared_error(y_test, y_pred)))

print("Predictor Columns : " + str(predictor_columns))
print("Linear Regression coefficients: " + str(lr.coef_))

# Normalize the coefficients
lr_coef = np.abs(lr.coef_)
lr_coef = lr_coef/np.sum(lr_coef)
# lr_coef is an array inside array, hence taking first element
plot_feature_importance_model(lr_coef[0], 'Linear Regression (normalized abs(beta))', ax1)
print("************************************************************")


# Decision Tree
print("******************** DECISION TREE ********************")

dt = DecisionTreeRegressor()
dt.fit(X_train, y_train)
y_pred = dt.predict(X_test)
print("R squared for the Decision Tree model: " + str(r2_score(y_test, y_pred)))
print("MSE for the Decision Tree model: " + str(mean_squared_error(y_test, y_pred)))

plot_feature_importance_model(dt.feature_importances_, 'Decision Tree', ax2)
perm = PermutationImportance(dt).fit(X_train, y_train)
# Negative values mean that a random permutation is better. Setting those to zero.
perm_feat_imp = perm.feature_importances_.copy()
perm_feat_imp[perm_feat_imp < 0] = 0
perm_feat_imp = perm_feat_imp/np.sum(perm_feat_imp)
plot_feature_importance_model(perm_feat_imp, 'Decision Tree (normed Permutation FI)', ax3)

print("************************************************************")



# Random Forest
print("******************** RANDOM FOREST ********************")

rf = RandomForestRegressor(n_estimators=10,
                           n_jobs=-1,
                           bootstrap=True,
                           random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
print("R squared for the Random Forest model: " + str(r2_score(y_test, y_pred)))
print("MSE for the Random Forest model: " + str(mean_squared_error(y_test, y_pred)))

plot_feature_importance_model(rf.feature_importances_, 'Random Forest', ax4)
perm = PermutationImportance(rf).fit(X_train, y_train)
# Negative values mean that a random permutation is better. Setting those to zero.
perm_feat_imp = perm.feature_importances_.copy()
perm_feat_imp[perm_feat_imp < 0] = 0
perm_feat_imp = perm_feat_imp/np.sum(perm_feat_imp)
plot_feature_importance_model(perm_feat_imp, 'Random Forest (normed Permutation FI)', ax5)

print("************************************************************")



# XG-BOOST
print("******************** XG-BOOST ********************")

xg_reg = xgb.XGBRegressor(objective='reg:squarederror', learning_rate = 0.1, max_depth = 5,
                          alpha = 10, n_estimators = 10, importance_type='gain')

xg_reg.fit(X_train, y_train)
y_pred = xg_reg.predict(X_test)
print("R squared for the XGBoost model: " + str(r2_score(y_test, y_pred)))
print("MSE for the XGBoost model: " + str(mean_squared_error(y_test, y_pred)))
plot_feature_importance_model(xg_reg.feature_importances_, 'XGBoost', ax6)

perm = PermutationImportance(xg_reg).fit(X_train, y_train)
# Negative values mean that a random permutation is better. Setting those to zero.
perm_feat_imp = perm.feature_importances_.copy()
perm_feat_imp[perm_feat_imp < 0] = 0
perm_feat_imp = perm_feat_imp/np.sum(perm_feat_imp)
plot_feature_importance_model(perm_feat_imp, 'XGBoost (normed Permutation FI)', ax7)

print("************************************************************")

fig.tight_layout()
plt.show()