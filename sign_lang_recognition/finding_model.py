import pickle
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns

# Učitavanje podataka
dict_ = pickle.load(open("./data.pickle", 'rb'))
data = np.array(dict_['data'])
labels = np.array(dict_['labels'])

# Label encoding
le = LabelEncoder()
labels = le.fit_transform(labels)

# Podela na trening i test skup
X_train, X_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, stratify=labels, shuffle=True, random_state=42
)

# Standardizacija
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Modeli i parametri
models_and_params = [
    ("LogisticRegression", LogisticRegression(), {
        "C": np.linspace(10, 100, 100),
        "solver": ["lbfgs", "newton-cg", "saga"]
    }),
    ("SVM", SVC(), {
        "C": np.linspace(20, 100, 100),
        "kernel": ["linear", "rbf", "poly"],
        "gamma": ["scale", "auto"]
    }),
    ("RandomForest", RandomForestClassifier(), {
        "n_estimators": np.arange(5, 150, 5),
        "max_depth": np.arange(5, 35),
        "min_samples_split": np.arange(2, 10)
    })
]

# Random pretraga hiperparametara (CV)
results = []
for name, model, params in models_and_params:
    print(f"\n--- Optimizacija modela: {name} ---")
    search = RandomizedSearchCV(
        model, params, n_iter=30, cv=3, scoring="accuracy",
        n_jobs=1, verbose=3
    )
    search.fit(X_train, y_train)

    print(f"{name} -> Najbolji parametri: {search.best_params_}")
    print(f"{name} -> Najbolji skor (CV): {search.best_score_:.4f}")

    results.append((name, search.best_score_, search.best_estimator_))

# Biramo model koji je najbolji prema VALIDACIONOM skoru iz CV-a
best_name, best_score, best_model = max(results, key=lambda x: x[1])
print(f"\n✅ Najbolji model na validacionom skupu: {best_name} ({best_score:.4f})")

# Evaluacija na test skupu
y_test_pred = best_model.predict(X_test)
test_acc = accuracy_score(y_test, y_test_pred)
print(f"📊 Test tačnost: {test_acc * 100:.2f}%")

# Matrica konfuzije za test skup
cm = confusion_matrix(y_test, y_test_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title(f"Matrica konfuzije — {best_name} (test skup)")
plt.xlabel("Predikcije")
plt.ylabel("Stvarne klase")
plt.tight_layout()
plt.show()

# Čuvanje modela i scaler-a
with open("model.pickle", "wb") as f:
    pickle.dump({'model': best_model, 'scaler': scaler, 'label_encoder': le}, f)

print("\n💾 Sačuvan najbolji model (validacioni skup) i scaler.")
