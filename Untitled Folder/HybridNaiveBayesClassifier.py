import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB, BernoulliNB, CategoricalNB
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer

class HybridNaiveBayesClassifier:
    def __init__(self, quantitative_features, binary_features, nominal_features):
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
        X_preprocessed = self.preprocessor.fit_transform(X)
        
         # Otteniamo i nomi delle colonne trasformate
        col_names = self.quantitative_features + self.binary_features + self.nominal_features

        # Convertiamo in DataFrame per facilitare l'indexing
        X_preprocessed_df = pd.DataFrame(X_preprocessed, columns=col_names)

        # Separare i dati preprocessati per tipo di feature
        X_continuous = X_preprocessed_df[self.quantitative_features]
        X_binary = X_preprocessed_df[self.binary_features]
        X_nominal = X_preprocessed_df[self.nominal_features]

        self.gaussian_nb.fit(X_continuous, y)
        self.bernoulli_nb.fit(X_binary, y)
        self.categorical_nb.fit(X_nominal, y)
    
    def predict(self, X):
        X_preprocessed = self.preprocessor.transform(X)
        
         # Otteniamo i nomi delle colonne trasformate
        col_names = self.quantitative_features + self.binary_features + self.nominal_features

        # Convertiamo in DataFrame per facilitare l'indexing
        X_preprocessed_df = pd.DataFrame(X_preprocessed, columns=col_names)

        # Separare i dati preprocessati per tipo di feature
        X_continuous = X_preprocessed_df[self.quantitative_features]
        X_binary = X_preprocessed_df[self.binary_features]
        X_nominal = X_preprocessed_df[self.nominal_features]

        # Predizioni separate dai modelli
        y_pred_continuous = self.gaussian_nb.predict_proba(X_continuous)
        y_pred_binary = self.bernoulli_nb.predict_proba(X_binary)
        y_pred_nominal = self.categorical_nb.predict_proba(X_nominal)

        # Moltiplicare le probabilità
#         prods = y_pred_continuous * y_pred_binary * y_pred_nominal
        
        # Restituire la classe con la probabilità massima
#         return np.argmax(prods, axis=1)

        # Sommare i log delle probabilità
        logsum = np.log(y_pred_continuous) + np.log(y_pred_binary) + np.log(y_pred_nominal)
        
        # Restituire la classe con la probabilità massima
        return np.argmax(logsum, axis=1)
    
    def predict_proba(self, X):
        X_preprocessed = self.preprocessor.transform(X)
        
         # Otteniamo i nomi delle colonne trasformate
        col_names = self.quantitative_features + self.binary_features + self.nominal_features

        # Convertiamo in DataFrame per facilitare l'indexing
        X_preprocessed_df = pd.DataFrame(X_preprocessed, columns=col_names)

        # Separare i dati preprocessati per tipo di feature
        X_continuous = X_preprocessed_df[self.quantitative_features]
        X_binary = X_preprocessed_df[self.binary_features]
        X_nominal = X_preprocessed_df[self.nominal_features]

        # Predizioni separate dai modelli
        y_pred_continuous = self.gaussian_nb.predict_proba(X_continuous)
        y_pred_binary = self.bernoulli_nb.predict_proba(X_binary)
        y_pred_nominal = self.categorical_nb.predict_proba(X_nominal)

        # Moltiplicare le probabilità
#         prods = y_pred_continuous * y_pred_binary * y_pred_nominal
        
        # Normalizzare le probabilità
#         return prods / np.sum(prods, axis=1, keepdims=True)

         # Sommare i log delle probabilità
        logsum = np.log(y_pred_continuous) + np.log(y_pred_binary) + np.log(y_pred_nominal)
        
        # Convertire log-probabilità a probabilità
        return np.exp(logsum) / np.sum(np.exp(logsum), axis=1, keepdims=True)

