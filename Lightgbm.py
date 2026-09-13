import Lightgbm as lgb
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

# Instantiate the classifier
LGBM_pipeline = lgb.LGBMClassifier()

# Build the pipeline
lgb_baseline_grid = Pipeline([
    ('scl', MinMaxScaler()),
    ('pca', PCA(n_components=45)),
    ('clf', LGBM_pipeline)
])

# Set grid search parameters
param_grid_lgb = {
    'learning_rate': [0.1, 0.2],  # =eta, smaller number makes model more robust by shrinking weights on each step
    'max_depth': [20, 40, 80],  # max depth of a tree, controls overfitting
    'min_child_weight': [40],  # minimum sum of weights of all observations required in a child, higher values reduce over-fitting
    'subsample': [0.6],  # the fraction of observations to be randomly sampled for each tree.
    'n_estimators': [50, 100],
}

grid_lgb = GridSearchCV(lgb_baseline_grid, param_grid_lgb, scoring='accuracy', cv=None, n_jobs=1)

# Assuming X_train and y_train are defined
# Train the model
grid_lgb.fit(X_train, y_train)
best_parameters = grid_lgb.best_params_
print('Grid Search found the following optimal parameters: ')
for param_name in sorted(best_parameters.keys()):
    print('%s: %r' % (param_name, best_parameters[param_name]))

training_preds = grid_lgb.predict(X_train)
test_preds = grid_lgb.predict(X_test)
training_accuracy = accuracy_score(y_train, training_preds)
test_accuracy = accuracy_score(y_test, test_preds)

# Print the training and validation accuracy of the optimal grid search result
print('')
print('Training Accuracy: {:.4}%'.format(training_accuracy * 100))
print('Validation accuracy: {:.4}%'.format(test_accuracy * 100))