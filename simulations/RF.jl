using DecisionTree

n, m = 10^3, 5
features = randn(n, m)
weights = rand(-2:2, m)
labels = features * weights


# train regression tree
model = build_tree(labels, features)
# apply learned model
apply_tree(model, [-0.9, 3.0, 5.1, 1.9, 0.0])
# run 3-fold cross validation, returns array of coefficients of determination (R^2)
n_folds = 3
r2 = nfoldCV_tree(labels, features, n_folds)

# set of regression parameters and respective default values
# pruning_purity: purity threshold used for post-pruning (default: 1.0, no pruning)
# max_depth: maximum depth of the decision tree (default: -1, no maximum)
# min_samples_leaf: the minimum number of samples each leaf needs to have (default: 5)
# min_samples_split: the minimum number of samples in needed for a split (default: 2)
# min_purity_increase: minimum purity needed for a split (default: 0.0)
# n_subfeatures: number of features to select at random (default: 0, keep all)
# keyword rng: the random number generator or seed to use (default Random.GLOBAL_RNG)
n_subfeatures = 0;
max_depth = -1;
min_samples_leaf = 5;
min_samples_split = 2;
min_purity_increase = 0.0;
pruning_purity = 1.0;
seed = 3;

model = build_tree(labels, features,
    n_subfeatures,
    max_depth,
    min_samples_leaf,
    min_samples_split,
    min_purity_increase;
    rng=seed)

r2 = nfoldCV_tree(labels, features,
    n_folds,
    pruning_purity,
    max_depth,
    min_samples_leaf,
    min_samples_split,
    min_purity_increase;
    verbose=true,
    rng=seed)


# train regression forest, using 2 random features, 10 trees,
# averaging of 5 samples per leaf, and 0.7 portion of samples per tree
model = build_forest(labels, features, 2, 10, 0.7, 5)
# apply learned model
apply_forest(model, [-0.9, 3.0, 5.1, 1.9, 0.0])
# run 3-fold cross validation on regression forest, using 2 random features per split
n_subfeatures = 2;
n_folds = 3;
r2 = nfoldCV_forest(labels, features, n_folds, n_subfeatures)

# set of regression build_forest() parameters and respective default values
# n_subfeatures: number of features to consider at random per split (default: -1, sqrt(# features))
# n_trees: number of trees to train (default: 10)
# partial_sampling: fraction of samples to train each tree on (default: 0.7)
# max_depth: maximum depth of the decision trees (default: no maximum)
# min_samples_leaf: the minimum number of samples each leaf needs to have (default: 5)
# min_samples_split: the minimum number of samples in needed for a split (default: 2)
# min_purity_increase: minimum purity needed for a split (default: 0.0)
# keyword rng: the random number generator or seed to use (default Random.GLOBAL_RNG)
#              multi-threaded forests must be seeded with an `Int`
n_subfeatures = -1;
n_trees = 10;
partial_sampling = 0.7;
max_depth = -1;
min_samples_leaf = 5;
min_samples_split = 2;
min_purity_increase = 0.0;
seed = 3;

model = build_forest(labels, features,
    n_subfeatures,
    n_trees,
    partial_sampling,
    max_depth,
    min_samples_leaf,
    min_samples_split,
    min_purity_increase;
    rng=seed)

r2 = nfoldCV_forest(labels, features,
    n_folds,
    n_subfeatures,
    n_trees,
    partial_sampling,
    max_depth,
    min_samples_leaf,
    min_samples_split,
    min_purity_increase;
    verbose=true,
    rng=seed)
