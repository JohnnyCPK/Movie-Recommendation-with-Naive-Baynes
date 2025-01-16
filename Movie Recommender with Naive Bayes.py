import numpy as np
import pandas as pd

# Specify the path to the ratings data file
data_path = r'C:\Users\johnp\Downloads\Movie Reommendation with Naive Baynes\ml-1m\ml-1m\ratings.dat'

# Load the ratings dataset using pandas
# `sep='::'` is used because the data uses '::' as a separator
# `engine='python'` ensures compatibility with custom separators
df = pd.read_csv(data_path, header=None, sep='::', engine='python')

# Add column names for better understanding of the dataset
df.columns = ['user_id', 'movie_id', 'rating', 'timestamp']

# Display the first few rows of the dataset to verify successful loading
print("First 5 rows of the dataset:\n", df.head())

# Count and display the number of unique users and movies in the dataset
n_users = df['user_id'].nunique()
n_movies = df['movie_id'].nunique()
print(f"Total number of users: {n_users}")
print(f"Total number of movies: {n_movies}")

# Function to create a user-item rating matrix and map movie IDs
def load_user_rating_data(df, n_users, n_movies):
    # Initialize a zero matrix of shape [n_users, n_movies] to hold ratings
    data = np.zeros([n_users, n_movies], dtype=np.intc)

    # Create a dictionary to map movie IDs to matrix column indices
    movie_id_mapping = {}

    # Populate the matrix with user ratings
    for user_id, movie_id, rating in zip(df['user_id'], df['movie_id'], df['rating']):
        user_id = int(user_id) - 1  # Convert user ID to zero-based indexing
        if movie_id not in movie_id_mapping:
            movie_id_mapping[movie_id] = len(movie_id_mapping)
        data[user_id, movie_id_mapping[movie_id]] = rating

    return data, movie_id_mapping

# Generate the user-item matrix and mapping
data, movie_id_mapping = load_user_rating_data(df, n_users, n_movies)

# Count and display the distribution of ratings in the dataset
values, counts = np.unique(data, return_counts=True)
print("Rating distribution:")
for value, count in zip(values, counts):
    print(f"  Rating {value}: {count} instances")

# Show the number of ratings per movie
print("\nNumber of ratings per movie (top 10):\n", df['movie_id'].value_counts().head(10))

# Select a target movie for recommendation analysis
# This movie ID is chosen as an example; it can be replaced with any valid ID
target_movie_id = 2858  # Example movie ID

# Create features (X) and labels (Y) for machine learning
# Remove the target movie column from the dataset
X_raw = np.delete(data, movie_id_mapping[target_movie_id], axis=1)

# Extract the target movie's ratings
Y_raw = data[:, movie_id_mapping[target_movie_id]]

# Filter out rows where the target movie has not been rated
X = X_raw[Y_raw > 0]
Y = Y_raw[Y_raw > 0]

# Display the shapes of the filtered feature matrix and labels
print(f"\nShape of X (feature matrix): {X.shape}")
print(f"Shape of Y (labels): {Y.shape}")

# Convert ratings to binary classification for recommendation
# Ratings greater than the threshold are considered positive
recommend = 3  # Ratings above this are considered positive
Y[Y <= recommend] = 0  # Negative class
Y[Y > recommend] = 1  # Positive class

# Count and display the number of positive and negative samples
n_pos = (Y == 1).sum()
n_neg = (Y == 0).sum()
print(f"\nNumber of positive samples: {n_pos}")
print(f"Number of negative samples: {n_neg}")

# Split the data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
print(f"\nTraining set size: {len(Y_train)}")
print(f"Testing set size: {len(Y_test)}")

# Train a Naive Bayes classifier on the training data
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB(alpha=1.0, fit_prior=True)
clf.fit(X_train, Y_train)

# Predict probabilities for the test set
prediction_prob = clf.predict_proba(X_test)
print("\nPredicted probabilities for the first 10 test samples:\n", prediction_prob[:10])

# Predict binary classes for the test set
prediction = clf.predict(X_test)
print("\nPredicted classes for the first 10 test samples:\n", prediction[:10])

# Calculate and display the accuracy of the classifier
accuracy = clf.score(X_test, Y_test)
print(f"\nAccuracy of the model: {accuracy*100:.2f}%")

# Display the confusion matrix
from sklearn.metrics import confusion_matrix
conf_matrix = confusion_matrix(Y_test, prediction, labels=[0, 1])
print("\nConfusion Matrix:\n", conf_matrix)

# Evaluate the classifier with precision, recall, and F1-score
from sklearn.metrics import precision_score, recall_score, f1_score
precision = precision_score(Y_test, prediction, pos_label=1)
recall = recall_score(Y_test, prediction, pos_label=1)
f1 = f1_score(Y_test, prediction, pos_label=1)
print(f"\nPrecision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")

# Generate a detailed classification report
from sklearn.metrics import classification_report
report = classification_report(Y_test, prediction)
print("\nClassification Report:\n", report)

# Compute the Receiver Operating Characteristic (ROC) curve
pos_prob = prediction_prob[:, 1]  # Probability of the positive class
thresholds = np.arange(0.0, 1.1, 0.05)  # Define thresholds from 0 to 1 in increments of 0.05
true_pos, false_pos = [0] * len(thresholds), [0] * len(thresholds)

# Calculate true positives and false positives for each threshold
for pred, y in zip(pos_prob, Y_test):
    for i, threshold in enumerate(thresholds):
        if pred >= threshold:  # Prediction is positive at current threshold
            if y == 1:
                true_pos[i] += 1  # True positive
            else:
                false_pos[i] += 1  # False positive
        else:
            break

# Normalize true positives and false positives to rates
n_pos_test = (Y_test == 1).sum()  # Total positives in the test set
n_neg_test = (Y_test == 0).sum()  # Total negatives in the test set
true_pos_rate = [tp / n_pos_test for tp in true_pos]
false_pos_rate = [fp / n_neg_test for fp in false_pos]

# Plot the ROC curve
import matplotlib.pyplot as plt
plt.figure()
lw = 2  # Line width for the plot
plt.plot(false_pos_rate, true_pos_rate, color='darkorange', lw=lw, label='ROC curve')
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.show()

# Compute and print the Area Under the Curve (AUC) score
from sklearn.metrics import roc_auc_score
auc_score = roc_auc_score(Y_test, pos_prob)
print(f"\nAUC Score: {auc_score:.2f}")

# Perform k-fold cross-validation to evaluate the model
from sklearn.model_selection import StratifiedKFold
k = 5  # Number of folds
k_fold = StratifiedKFold(n_splits=k, random_state=42, shuffle=True)

# Define hyperparameter options for cross-validation
smoothing_factor_option = [1, 2, 3, 4, 5, 6]  # Different smoothing factors to test
fit_prior_option = [True, False]  # Whether to use class priors or assume uniform priors
auc_record = {}  # Dictionary to store AUC scores for each hyperparameter combination

# Perform k-fold cross-validation
for train_indices, test_indices in k_fold.split(X, Y):
    # Split the data into training and testing sets for the current fold
    X_train_k, X_test_k = X[train_indices], X[test_indices]
    Y_train_k, Y_test_k = Y[train_indices], Y[test_indices]

    for alpha in smoothing_factor_option:
        # Initialize storage for results of this alpha if not already present
        if alpha not in auc_record:
            auc_record[alpha] = {}

        for fit_prior in fit_prior_option:
            # Train a Naive Bayes model with the current parameters
            clf = MultinomialNB(alpha=alpha, fit_prior=fit_prior)
            clf.fit(X_train_k, Y_train_k)

            # Predict probabilities for the test set
            prediction_prob = clf.predict_proba(X_test_k)
            pos_prob = prediction_prob[:, 1]  # Extract positive class probabilities

            # Calculate AUC for the current fold and parameters
            auc = roc_auc_score(Y_test_k, pos_prob)

            # Accumulate AUC scores for averaging across folds
            auc_record[alpha][fit_prior] = auc + auc_record[alpha].get(fit_prior, 0.0)

# Display the cross-validation results
print("\nCross-Validation Results (AUC Scores):")
print("Smoothing | Fit Prior | AUC")
for smoothing, smoothing_record in auc_record.items():
    for fit_prior, auc in smoothing_record.items():
        # Print averaged AUC for each hyperparameter combination
        print(f"   {smoothing:<9} {fit_prior:<10} {auc / k:.5f}")

# Train the best model based on cross-validation results
clf = MultinomialNB(alpha=2.0, fit_prior=False)  # Example: best parameters identified from results
clf.fit(X_train, Y_train)  # Train the final model
pos_prob = clf.predict_proba(X_test)[:, 1]  # Predict probabilities on the test set
print(f"\nAUC with the best model: {roc_auc_score(Y_test, pos_prob):.2f}")
