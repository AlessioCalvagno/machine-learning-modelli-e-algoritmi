import numpy as np
from sklearn.naive_bayes import GaussianNB, BernoulliNB, CategoricalNB
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

class HybridNaiveBayesClassifier:
    def __init__(self, continuous_features, binary_features, nominal_features):
        self.continuous_features = continuous_features
        self.binary_features = binary_features
        self.nominal_features = nominal_features
        
        self.gaussian_nb = GaussianNB()
        self.bernoulli_nb = BernoulliNB()
        self.categorical_nb = CategoricalNB()
        
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), continuous_features),
                ('bin', 'passthrough', binary_features),
                ('cat', 'passthrough', nominal_features)
            ]
        )
    
    def fit(self, X, y):
        X_preprocessed = self.preprocessor.fit_transform(X)

        # Separare i dati preprocessati per tipo di feature
        X_continuous = X_preprocessed[:, :len(self.continuous_features)]
        X_binary = X_preprocessed[:, len(self.continuous_features):len(self.continuous_features) + len(self.binary_features)]
        X_nominal = X_preprocessed[:, len(self.continuous_features) + len(self.binary_features):]

        self.gaussian_nb.fit(X_continuous, y)
        self.bernoulli_nb.fit(X_binary, y)
        self.categorical_nb.fit(X_nominal, y)
    
    def predict(self, X):
        X_preprocessed = self.preprocessor.transform(X)

        # Separare i dati preprocessati per tipo di feature
        X_continuous = X_preprocessed[:, :len(self.continuous_features)]
        X_binary = X_preprocessed[:, len(self.continuous_features):len(self.continuous_features) + len(self.binary_features)]
        X_nominal = X_preprocessed[:, len(self.continuous_features) + len(self.binary_features):]

        # Predizioni separate dai modelli
        y_pred_continuous = self.gaussian_nb.predict_proba(X_continuous)
        y_pred_binary = self.bernoulli_nb.predict_proba(X_binary)
        y_pred_nominal = self.categorical_nb.predict_proba(X_nominal)

        # Moltiplicare le probabilità
        prods = y_pred_continuous * y_pred_binary * y_pred_nominal
        
        # Restituire la classe con la probabilità massima
        return np.argmax(prods, axis=1)
    
    def predict_proba(self, X):
        X_preprocessed = self.preprocessor.transform(X)

        # Separare i dati preprocessati per tipo di feature
        X_continuous = X_preprocessed[:, :len(self.continuous_features)]
        X_binary = X_preprocessed[:, len(self.continuous_features):len(self.continuous_features) + len(self.binary_features)]
        X_nominal = X_preprocessed[:, len(self.continuous_features) + len(self.binary_features):]

        # Predizioni separate dai modelli
        y_pred_continuous = self.gaussian_nb.predict_proba(X_continuous)
        y_pred_binary = self.bernoulli_nb.predict_proba(X_binary)
        y_pred_nominal = self.categorical_nb.predict_proba(X_nominal)

        # Moltiplicare le probabilità
        prods = y_pred_continuous * y_pred_binary * y_pred_nominal
        
        # Normalizzare le probabilità
        return prods / np.sum(prods, axis=1, keepdims=True)

    
###################################################
# Esempio di utilizzo
# Supponiamo che il tuo dataset abbia le seguenti caratteristiche
# df è il tuo dataframe pandas
# target è il nome della colonna target

# Dividi il dataset nelle sue feature e nel target
X = df.drop(columns=['target'])
y = df['target']

# Identifica i tipi di feature
continuous_features = ['feature1', 'feature2']  # sostituisci con i nomi delle tue feature continue
binary_features = ['feature3', 'feature4']     # sostituisci con i nomi delle tue feature binarie
nominal_features = ['feature5', 'feature6']    # sostituisci con i nomi delle tue feature nominali

# Inizializza il classificatore
hybrid_nb = HybridNaiveBayesClassifier(continuous_features, binary_features, nominal_features)

# Dividi il dataset in training e test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Adatta il modello combinato sui dati di training
hybrid_nb.fit(X_train, y_train)

# Fai predizioni sui dati di test
y_pred = hybrid_nb.predict(X_test)

# Valuta il modello
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')


 Generato da ProfAI - https://prof.profession.ai/