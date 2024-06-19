import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB, BernoulliNB, CategoricalNB
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer

class HybridNaiveBayesClassifier:
    """
    This class implements a Hybrid Naive Bayes classifier that combines
    Gaussian Naive Bayes, Bernoulli Naive Bayes, and Categorical Naive Bayes
    for classification tasks.
    """
    
    def __init__(self, quantitative_features, binary_features, nominal_features):
        """
        This function initializes the classifier with the following parameters:

        - quantitative_features: A list of column names containing quantitative data.
        - binary_features: A list of column names containing binary data.
        - nominal_features: A list of column names containing nominal data.
        """
        self.quantitative_features = quantitative_features
        self.binary_features = binary_features
        self.nominal_features = nominal_features
        
        self.gaussian_nb = GaussianNB()
        self.bernoulli_nb = BernoulliNB()
        self.categorical_nb = CategoricalNB()
        
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), quantitative_features),
                ('bin', OrdinalEncoder(handle_unknown="use_encoded_value",
                                       unknown_value=np.nan), binary_features),
                ('cat', OrdinalEncoder(handle_unknown="use_encoded_value",
                                      unknown_value=np.nan), nominal_features)
            ],
            remainder="passthrough"
        )
    
    def fit(self, X, y):
        """
        This function fits the classifier to the training data.

        - X: The training data as a pandas DataFrame.
        - y: The target labels as a pandas Series or NumPy array.
        """
        X_preprocessed = self.preprocessor.fit_transform(X)
        
         # Get the names of the transformed columns
        col_names = self.quantitative_features + self.binary_features + self.nominal_features

        # Convert to DataFrame for easier indexing
        X_preprocessed_df = pd.DataFrame(X_preprocessed, columns=col_names)

        # Separate preprocessed data by feature type
        X_continuous = X_preprocessed_df[self.quantitative_features]
        X_binary = X_preprocessed_df[self.binary_features]
        X_nominal = X_preprocessed_df[self.nominal_features]

        self.gaussian_nb.fit(X_continuous, y)
        self.bernoulli_nb.fit(X_binary, y)
        self.categorical_nb.fit(X_nominal, y)
    
    def predict(self, X):
        """
        This function predicts the class labels for new data points.

        - X: The data to predict on as a pandas DataFrame.

        Returns:
          A NumPy array containing the predicted class labels.
        """
        X_preprocessed = self.preprocessor.transform(X)
        
         # Get the names of the transformed columns
        col_names = self.quantitative_features + self.binary_features + self.nominal_features

        # Convert to DataFrame for easier indexing
        X_preprocessed_df = pd.DataFrame(X_preprocessed, columns=col_names)

        # Separate preprocessed data by feature type
        X_continuous = X_preprocessed_df[self.quantitative_features]
        X_binary = X_preprocessed_df[self.binary_features]
        X_nominal = X_preprocessed_df[self.nominal_features]

        #  Separate log probabilities from each model
        y_log_prob_continuous = self.gaussian_nb.predict_log_proba(X_continuous)
        y_log_prob_binary = self.bernoulli_nb.predict_log_proba(X_binary)
        y_log_prob_nominal = self.categorical_nb.predict_log_proba(X_nominal)

        # Sum the log probabilities
        logsum = y_log_prob_continuous + y_log_prob_binary + y_log_prob_nominal
        
        # Return the class with the maximum probability
        return np.argmax(logsum, axis=1)
    
    def predict_proba(self, X):
         """
        This function predicts the probability of each class for new data points.
        - X: The data to predict on as a pandas DataFrame.

        Returns:
          A NumPy array containing the predicted probabilities for each class.
        """
        X_preprocessed = self.preprocessor.transform(X)
        
         # Get the names of the transformed columns
        col_names = self.quantitative_features + self.binary_features + self.nominal_features

        # Convert to DataFrame for easier indexing
        X_preprocessed_df = pd.DataFrame(X_preprocessed, columns=col_names)

        # Separate preprocessed data by feature type
        X_continuous = X_preprocessed_df[self.quantitative_features]
        X_binary = X_preprocessed_df[self.binary_features]
        X_nominal = X_preprocessed_df[self.nominal_features]

        #  Separate log probabilities from each model
        y_log_prob_continuous = self.gaussian_nb.predict_log_proba(X_continuous)
        y_log_prob_binary = self.bernoulli_nb.predict_log_proba(X_binary)
        y_log_prob_nominal = self.categorical_nb.predict_log_proba(X_nominal)

         # Sum the log probabilities
        logsum = y_log_prob_continuous + y_log_prob_binary + y_log_prob_nominal
        
        # Convert log probabilities to probabilities
        return np.exp(logsum) / np.sum(np.exp(logsum), axis=1, keepdims=True)

