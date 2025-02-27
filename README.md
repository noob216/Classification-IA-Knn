# Classification-IA-Knn

## Introduction

Ce projet implémente une version optimisée de l'algorithme des k plus proches voisins (KNN), en intégrant des prétraitements avancés, une stratégie d'augmentation des données (Borderline-SMOTE), ainsi qu'une recherche étendue d'hyperparamètres.

L'objectif est de fournir une classification robuste et efficace, tout en exploitant des techniques avancées d'ingénierie des caractéristiques et d'équilibrage des données.

## Fonctionnalités Principales

- **Normalisation standard** : Optimisation du traitement des données pour une convergence plus stable.
- **Génération de caractéristiques polynomiales** : Capture des relations complexes entre les variables.
- **Borderline-SMOTE** : Suréchantillonnage des classes minoritaires tout en conservant les frontières de décision.
- **Implémentation manuelle de KNN** : Suppression de toute dépendance à scikit-learn pour plus de contrôle.
- **Optimisation des hyperparamètres** : Recherche extensive des meilleurs paramètres pour maximiser la précision.

## Installation

### Prérequis

- Python 3.x
- Bibliothèques :
  - `numpy`
  - `pandas`
  - `scipy`

Vous pouvez les installer avec :

```bash
pip install numpy pandas scipy
```

## Utilisation

### 1. Chargement des Données

Les données doivent être fournies sous forme de fichiers CSV :

- `train.csv` : Contient les données d'entraînement avec la colonne `Label`.
- `test.csv` : Contient les données de test sans labels.

Le script charge et traite ces fichiers automatiquement.

### 2. Prétraitement

- Normalisation des données.
- Génération de caractéristiques polynomiales.
- Application de Borderline-SMOTE pour équilibrer les classes.

### 3. Entraîment du Modèle

L'algorithme KNN est implémenté avec une recherche de paramètres incluant :

- Nombre de voisins (`n_neighbors` entre 1 et 25).
- Type de pondération (`uniform` vs `distance`).
- Métrique de distance (Manhattan).

Le modèle optimal est sélectionné en fonction du meilleur taux de précision sur les données d'entraîment.

### 4. Prédiction et Sauvegarde des Résultats

Une fois le modèle optimisé, les prédictions sont générées sur le jeu de test et enregistrées dans un fichier CSV :

```
resultats_final_tuned_v6_borderline_smote.csv
```

## Structure du Code

- **standard\_scaler(X)** : Normalise les données.
- **polynomial\_features(X, degree=2, cross\_terms=True)** : Crée des interactions polynomiales entre variables.
- **borderline\_smote(X, y, k=5, perturbation\_factor=0.02)** : Augmente les données minoritaires en préservant les frontières des classes.
- **KNNClassifier(n\_neighbors, weights, metric)** : Implémente KNN avec une distance Manhattan et gestion des pondérations.
- **CustomPipeline(steps)** : Gère le prétraitement des données sous forme de pipeline modulaire.
- **Recherche des hyperparamètres** : Parcours un espace de paramètres optimisé pour KNN.

## Résultats Attendus

- Amélioration de la précision grâce à l'ingénierie des caractéristiques.
- Meilleure gestion du déséquilibre des classes.
- Modèle KNN plus performant grâce à une recherche optimisée d'hyperparamètre.

## Conclusion

Ce projet démontre comment un algorithme simple comme KNN peut être amélioré significativement avec des techniques avancées de prétraitement et d'optimisation. Il offre un modèle performant tout en restant compréhensible et adaptable à différents jeux de données.

---

**Auteur** : ALI Mathis
