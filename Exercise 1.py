import numpy as np
from sklearn.naive_bayes import BernoulliNB

# Training data (binary features)
X_train = np.array([
    [0,1,1],  # Example 1
    [0,0,1],  # Example 2
    [0,0,0],  # Example 3
    [1,1,0]   # Example 4
])

# Corresponding labels for training data
Y_train = ['Y', 'N', 'Y', 'Y']

# Test data to classify
X_test = np.array([[1,1,0]])  # Example to classify

# Function to group indices of training examples by their label
def get_label_indices(labels):
    from collections import defaultdict
    label_indices = defaultdict(list)
    for index, label in enumerate(labels):
        label_indices[label].append(index)  # Map each label to its indices
    return label_indices

# Group indices of the training samples by their labels
label_indices = get_label_indices(Y_train)
print('\nLabel Indices:')
for label, indices in label_indices.items():
    print(f'  {label}: {indices}')

# Function to calculate the prior probabilities of each label
def get_prior(label_indices):
    # Count occurrences of each label
    prior = {label: len(indices) for label, indices in label_indices.items()}
    total_count = sum(prior.values())  # Total number of training samples
    for label in prior:
        prior[label] /= total_count  # Normalize counts to calculate probabilities
    return prior

# Calculate prior probabilities of labels
prior = get_prior(label_indices)
print('\nPrior Probabilities:')
for label, prob in prior.items():
    print(f'  {label}: {prob:.4f}')

# Function to calculate likelihood of features given each label
def get_likelihood(features, label_indices, smoothing=0):
    likelihood = {}
    for label, indices in label_indices.items():
        # Sum feature occurrences for all examples with the current label
        likelihood[label] = features[indices, :].sum(axis=0) + smoothing
        total_count = len(indices)  # Count of examples with this label
        # Normalize and account for smoothing
        likelihood[label] = likelihood[label] / (total_count + 2 * smoothing)
    return likelihood

# Additive smoothing value to avoid zero probabilities
smoothing = 1
likelihood = get_likelihood(X_train, label_indices, smoothing)
print('\nLikelihoods:')
for label, probs in likelihood.items():
    print(f'  {label}: {probs}')

# Function to compute posterior probabilities for each class given the test data
def get_posterior(X, prior, likelihood):
    posteriors = []
    for x in X: 
        posterior = prior.copy()  # Start with prior probabilities
        for label, likelihood_label in likelihood.items():
            for index, bool_value in enumerate(x):
                # Update posterior by multiplying likelihoods
                posterior[label] *= likelihood_label[index] if bool_value else (1 - likelihood_label[index])
        
        # Normalize posteriors so they sum to 1
        sum_posterior = sum(posterior.values())
        for label in posterior:
            if posterior[label] == float('inf'):
                posterior[label] = 1.0  # Handle numerical instability
            else:
                posterior[label] /= sum_posterior
        posteriors.append(posterior.copy())  # Store posterior for this test sample
    return posteriors

# Calculate posterior probabilities for the test data
posterior = get_posterior(X_test, prior, likelihood)
print('\nPosterior Probabilities for Test Data:')
for idx, post in enumerate(posterior):
    print(f'  Test Example {idx + 1}: {post}')

# Use scikit-learn's Bernoulli Naive Bayes for comparison
clf = BernoulliNB(alpha=1.0, fit_prior=True)  # Alpha=1 applies Laplace smoothing
clf.fit(X_train, Y_train)  # Train the model
pred_prob = clf.predict_proba(X_test)  # Predict class probabilities for the test data
print('\n[Scikit-Learn] Predicted Probabilities:')
for idx, probs in enumerate(pred_prob):
    print(f'  Test Example {idx + 1}: {probs}')