import pickle
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

dict_ = pickle.load(open("./data.pickle", 'rb'))    # Učitavanje podataka

data = np.array(dict_['data'])
labels = np.array(dict_['labels'])

print(np.size(data))

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, stratify=labels, shuffle=True)    # Podela na trening i test skup

scaler = StandardScaler()    #  Standardizacija podataka
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = SVC(kernel="rbf", gamma="scale", C=100)    # Optimalan model dobijen pomoću cross validacije
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(f'{accuracy_score(y_pred, y_test)*100:.2f}% accuracy!')    # Tačnost na test skupu

dict_letters = {
    str(i + 1): letter
    for i, letter in enumerate(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
                                'K', 'L', 'M', 'N', 'O', 'P', 'R', 'S', 'T', 'U', 'V', 'Z'])
}

unique_labels = sorted(np.unique(labels), key=lambda x: int(x))
letters = [dict_letters[str(i)] for i in unique_labels]

cm = confusion_matrix(y_test, y_pred, labels=unique_labels)    # Matrica konfuzije za podatke na test skupu

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=letters, yticklabels=letters,
            cbar=False, linewidths=0.5, square=True)
plt.xlabel('Predikcije')
plt.ylabel('Stvarne klase')
plt.title('Matrica konfuzije (test skup)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D  
import matplotlib.pyplot as plt
import numpy as np

# PCA za 3 komponente
pca = PCA(n_components=3)
X_proj = pca.fit_transform(X_test)

colors = plt.cm.tab20(np.linspace(0, 1, len(np.unique(y_test))))

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

for i, cls in enumerate(np.unique(y_test)):
    idx = np.where(y_test == cls)
    ax.scatter(
        X_proj[idx, 0],
        X_proj[idx, 1],
        X_proj[idx, 2],
        color=colors[i],
        label=dict_letters[str(cls)],
        s=60,
        alpha=0.8,
        edgecolor='black',
        linewidth=0.4
    )

ax.set_xlabel("PCA komponenta 1")
ax.set_ylabel("PCA komponenta 2")
ax.set_zlabel("PCA komponenta 3")
ax.set_title("SVM – PCA projekcija test skupa", fontsize=13, fontweight='bold')
ax.legend(title="Klasa", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

with open("model.pickle", "wb") as f:
    pickle.dump({'model': model, 'scaler': scaler}, f)
