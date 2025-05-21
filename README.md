# DL_challenge_classification_dogs_cats
# **Classification d'images : Chats vs Chiens avec CNN**

## **Description du projet**
Ce projet utilise un **réseau de neurones convolutifs (CNN)** pour classifier des images de chats et de chiens.  
Le modèle prend en entrée des images de taille **128x128 pixels** et prédit si elles représentent un **chat (classe 0)** ou un **chien (classe 1)**.

## **Technologies utilisées**
- **Python**  
- **TensorFlow / Keras**  
- **NumPy**  
- **Matplotlib**  
- **Sklearn (pour la matrice de confusion)**  

## **Structure du projet**


deep_learning/ │── data/ │ ├── pets/ │ │ ├── train/ 
### Dossier des images d'entraînement (chats et chiens) │ │ ├── test/ 
### Dossier des images de test (chats et chiens) │── model.py  
### Fichier contenant la définition du modèle CNN │── train.py  
### Script pour entraîner le modèle │── test.py  
### Script pour tester le modèle et afficher les résultats │── README.md  
### Documentation du projet


## **Installation et configuration**
1. **Cloner le projet :**  
   ```bash
   git clone https://github.com/elvis-messiaen/dog_cat_deep_learning.git
   cd dog_cat_deep_learning

## Créer un environnement virtuel :
python -m venv .venv
source .venv/bin/activate  # Sur macOS/Linux
.venv\Scripts\activate     # Sur Windows

## Installer les dépendances :
pip install -r requirements.txt

## Modèle CNN
Le modèle est composé de quatre couches suivies de max-pooling, puis de couches entièrement connectées :

```python
model = Sequential([
    Input(shape=(128, 128, 3)),  
    Conv2D(32, (3,3), activation='relu', padding="same"),
    MaxPooling2D(pool_size=(2,2)),

    Conv2D(64, (3,3), activation='relu', padding="same"),
    MaxPooling2D(pool_size=(2,2)),

    Conv2D(128, (3,3), activation='relu', padding="same"),
    MaxPooling2D(pool_size=(2,2)),

    Conv2D(256, (3,3), activation='relu', padding="same"),
    MaxPooling2D(pool_size=(2,2)),

    Flatten(),
    Dense(512, activation='relu', kernel_regularizer=l2(0.01)),
    Dropout(0.5),
    Dense(256, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])
```

## Entraînement du modèle
L'entraînement s'effectue sur 10 époques avec un batch de 32 images :
```python
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=32)
```

## Évaluation et Prédictions
Une fois entraîné, le modèle est évalué sur les images de test avec l'affichage des performances et des prédictions :

test_loss, test_accuracy = model.evaluate(X_test, y_test, batch_size=64)
print(f"Loss sur les données de test : {test_loss}")
print(f"Accuracy sur les données de test : {test_accuracy}")

## Affichage des résultats avec Matplotlib
```python
fig, axes = plt.subplots(3, 3, figsize=(10, 10))
axes = axes.ravel()
for i in range(9):
    axes[i].imshow(X_test[i])
    axes[i].axis("off")
    label = "Chien" if y_pred_classes[i] == 1 else "Chat"
    true_label = "Chien" if y_test[i] == 1 else "Chat"
    axes[i].set_title(f"Prédit : {label} / Réel : {true_label}")
plt.show()
```

## Optimisations possibles

#### Augmentation des données (Data Augmentation) pour améliorer la généralisation.
#### Réduction du learning rate pour une meilleure convergence.
#### Augmentation du nombre d’époques pour un meilleur apprentissage.
#### Utilisation de modèles pré-entraînés (VGG16, ResNet) pour améliorer la précision.