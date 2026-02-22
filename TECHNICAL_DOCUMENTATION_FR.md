# Documentation Technique - MyTorch

## 1. Vue d'Ensemble du Projet

MyTorch est un framework de réseaux de neurones personnalisé écrit en C++20 (STL), conçu spécifiquement pour l'analyse de positions d'échecs. Il implémente un réseau de neurones dense entièrement fonctionnel *from scratch*, sans dépendances externes de machine learning comme PyTorch ou TensorFlow.

Le cas d'utilisation principal est `my_torch_analyzer`, un outil en ligne de commande qui entraîne un réseau à prédire l'état d'une partie d'échecs (Rien, Échec, Échec et Mat) ou la couleur gagnante, basé sur une chaîne FEN (Forsyth-Edwards Notation).

## 2. Architecture du Système

Le projet est divisé en deux couches logiques distinctes :

### 2.1 Librairie Neuronale (src/nn)
Ce module indépendant gère les opérations mathématiques du réseau de neurones.

*   **Network (Réseau)** : Le conteneur de haut niveau qui gère une séquence de couches. Il orchestre les passes avant (forward) et arrière (backward).
*   **Layer (Couche)** : Représente une couche dense (entièrement connectée). Elle contient les poids, les biais et les accumulateurs de gradients. Elle effectue la multiplication matricielle `Y = Activation(WX + B)`.
*   **Activations** : Une classe utilitaire statique fournissant les fonctions d'activation (Sigmoid, ReLU) et leurs dérivées.
*   **Loss (Perte)** : Fournit les fonctions de coût (CrossEntropy) pour évaluer la performance du modèle et calculer les gradients.

### 2.2 Application Analyzer (src/analyzer)
Ce module implémente la logique métier spécifique à l'analyse d'échecs.

*   **CLI** : Le point d'entrée de l'interface en ligne de commande. Il gère l'analyse des arguments, le chargement de la configuration et pilote les flux de travail d'entraînement et de prédiction.
*   **FENParser** : Un parseur optimisé qui convertit une chaîne FEN en un vecteur d'entrée normalisé de taille 838.
*   **Dataset** : Gère le chargement et l'analyse des jeux de données CSV en mémoire, y compris le mappage des étiquettes (labels).

## 3. Détails d'Implémentation

### 3.1 Logique du Réseau de Neurones

#### Passe Avant (Forward)
La propagation est séquentielle. Un vecteur d'entrée entre dans la première couche. La sortie de la couche `N` devient l'entrée de la couche `N+1`.
Équation : `Z = Poids * Entrée + Biais`
Sortie : `A = Activation(Z)`

#### Passe Arrière (Backward / Rétropropagation)
L'apprentissage est réalisé via la Différentiation Automatique en Mode Inverse (Reverse Mode Differentiation).
1.  **Dérivée de la Loss** : On calcule le gradient de la fonction de perte par rapport à la sortie du réseau.
2.  **Propagation** : Chaque couche calcule le gradient par rapport à ses entrées (pour le passer à la couche précédente) et calcule localement les gradients par rapport à ses poids et biais.
3.  **Accumulation de Gradients** : Pour supporter l'entraînement par Mini-Batch, les gradients ne sont pas appliqués immédiatement. Ils sont sommés dans les structures `grad_weights_sum` et `grad_biases_sum` au sein de chaque couche.

#### Optimisation
Nous utilisons la Descente de Gradient Stochastique (SGD) avec support Mini-Batch et Décroissance du Taux d'Apprentissage (Learning Rate Decay).
*   **Mise à jour des Poids** : `NouveauPoids = AncienPoids - (TauxApprentissage * GradientAccumulé / TailleBatch)`
*   **Scheduler** : Le taux d'apprentissage est multiplié par un facteur `lr_decay` toutes les `decay_step` époques pour affiner la convergence.

### 3.2 Encodage des Entrées Échecs (FEN)

Une chaîne FEN brute est convertie en un vecteur plat de 838 valeurs flottantes pour servir d'entrée au réseau.

**Structure :**
*   **Plateau (64 cases x 13 canaux)** : 832 caractéristiques.
    *   Chaque case utilise un encodage One-Hot pour le type de pièce (Pion, Cavalier, Fou, Tour, Reine, Roi - Blanc/Noir) ou Vide.
*   **État du Jeu (6 caractéristiques)** :
    *   1 caractéristique pour le trait (Blanc=1, Noir=0).
    *   4 caractéristiques pour les droits de roque (KQkq).
    *   1 caractéristique pour la prise en passant (booléen).

## 4. Guide Développeur

### 4.1 Système de Build
Le projet utilise un Makefile standard.

*   **compiler** : `make`
*   **nettoyer** : `make clean`
*   **nettoyage complet** : `make fclean`
*   **exécution** : `./my_torch_analyzer`

### 4.2 Format de Configuration
Le support est strictement de style clé-valeur `.ini`.

```ini
layers=838,128,64,3     # Définition de la topologie
learning_rate=0.01      # Taux d'apprentissage initial
epochs=50               # Durée de l'entraînement
batch_size=32           # Taille d'accumulation des gradients
validation_ratio=0.2    # % de données gardées pour la validation
lr_decay=0.9            # Facteur appliqué au taux d'apprentissage
decay_step=10           # Intervalle d'époques pour la décroissance
```

### 4.3 Étendre le Framework

#### Ajouter une nouvelle Fonction d'Activation
1.  Modifier `include/Activations.hpp` pour déclarer la fonction et sa dérivée.
2.  Implémenter la logique dans `src/nn/Activations.cpp`.
3.  Ajouter la nouvelle valeur d'enum à `ActivationType` dans `Layer.hpp`.
4.  Mettre à jour les `switch cases` dans `Layer::forward` et `Layer::backward`.

#### Modifier le Format d'Entrée
La classe `FENParser` est isolée. Vous pouvez remplacer l'implémentation de `fenToVector` pour changer la représentation des positions d'échecs sans toucher au cœur du réseau de neurones.

## 5. Considérations de Performance
*   **Mémoire** : Le dataset est chargé entièrement en RAM pour la rapidité. Pour des datasets massifs (>10Go), une approche par itérateur de flux (streaming) serait requise dans `Dataset.cpp`.
*   **Maths** : Les opérations sont des boucles scalaires. Des optimisations SIMD ou une intégration BLAS pourraient accélérer significativement les multiplications matricielles.

## 6. Tests
Les tests sont situés dans le répertoire `tests/` et utilisent un header de test unitaire minimaliste personnalisé `unit_test.hpp`.

*   **Lancer** : `make tests`
*   **Portée** : Les tests couvrent la logique des composants (Layer, Activation, Network) et l'intégration (convergence XOR).
