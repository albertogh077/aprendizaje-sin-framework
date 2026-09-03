"""Implementación de Random Forest para clasificación binaria, desde cero."""
import time
import numpy as np
from collections import Counter
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support


class Node:
    """Nodo de un árbol de decisión.

    Un nodo es interno (de split) si tiene ``feature_idx`` y ``threshold``
    definidos, o una hoja si tiene ``value`` definido (clase predicha).

    Args:
        feature_idx (int, optional): Índice de la columna usada para el split.
        threshold (float, optional): Valor umbral del split (izquierda si
            ``x[feature_idx] <= threshold``).
        left (Node, optional): Subárbol izquierdo.
        right (Node, optional): Subárbol derecho.
        value (optional): Clase predicha, presente solo en nodos hoja.
    """
    def __init__(self, feature_idx=None, threshold=None, left=None, right=None, value=None):
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value


class RandomForest:
    """Random Forest Classifier - Ensamblaje de árboles de decisión.

    Implementación educativa (no vectorizada a nivel de árboles) de un
    clasificador Random Forest binario/multiclase basado en entropía y
    ganancia de información.

    Args:
        n_trees (int): Número de árboles a entrenar.
        max_depth (int): Profundidad máxima permitida por árbol.
        min_samples_split (int): Número mínimo de muestras requerido para
            seguir dividiendo un nodo.

    Attributes:
        trees (list[Node]): Raíces de los árboles ya entrenados.
    """

    def __init__(self, n_trees=10, max_depth=5, min_samples_split=5):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.trees = []

    def fit(self, X, y):
        """Entrena el bosque construyendo ``n_trees`` árboles independientes.

        Cada árbol se entrena sobre una muestra bootstrap (muestreo con
        reemplazo del mismo tamaño que el dataset original), lo que
        introduce diversidad entre árboles y es la base del "bagging".

        Args:
            X (np.ndarray): Matriz de características, shape (n_samples, n_features).
            y (np.ndarray): Vector de etiquetas, shape (n_samples,).
        """
        n_samples = X.shape[0]
        print(f"Entrenando Random Forest ({self.n_trees} árboles, max_depth={self.max_depth})...")
        start = time.time()

        for i in range(self.n_trees):
            # Bootstrap sampling: muestreo con reemplazo del mismo tamaño
            # que el dataset original, para que cada árbol vea una vista
            # ligeramente distinta de los datos.
            idxs = np.random.choice(n_samples, n_samples, replace=True)
            X_boot = X[idxs]
            y_boot = y[idxs]

            # Construir árbol
            tree = self._build_tree(X_boot, y_boot, depth=0)
            self.trees.append(tree)

            print(f"  Árbol {i + 1}/{self.n_trees} entrenado", flush=True)

        elapsed = time.time() - start
        print(f"Entrenamiento completo en {elapsed:.2f}s")

    def _build_tree(self, X, y, depth):
        """Construye un árbol de decisión recursivamente (algoritmo CART simplificado).

        En cada nodo se evalúan solo sqrt(n_features) características
        elegidas al azar (feature subsampling), lo que descorrelaciona los
        árboles del bosque y mejora la generalización del ensamblaje.

        Args:
            X (np.ndarray): Subconjunto de características del nodo actual.
            y (np.ndarray): Subconjunto de etiquetas del nodo actual.
            depth (int): Profundidad actual del nodo dentro del árbol.

        Returns:
            Node: Nodo hoja (con ``value``) o nodo interno con sus hijos.
        """
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))

        # Criterios de parada: profundidad máxima alcanzada, nodo puro
        # (una sola clase) o muy pocas muestras para seguir dividiendo.
        if depth >= self.max_depth or n_classes == 1 or n_samples < self.min_samples_split:
            return Node(value=self._most_common(y))

        # Seleccionar sqrt(n_features) características al azar
        n_feat_split = max(1, int(np.sqrt(n_features)))
        feat_idxs = np.random.choice(n_features, n_feat_split, replace=False)

        # Encontrar mejor split
        best_feat, best_thresh = self._best_split(X, y, feat_idxs)

        if best_feat is None:
            return Node(value=self._most_common(y))

        # Dividir y recursar
        left_mask = X[:, best_feat] <= best_thresh
        right_mask = ~left_mask

        left = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        return Node(feature_idx=best_feat, threshold=best_thresh, left=left, right=right)

    def _best_split(self, X, y, feat_idxs):
        """Busca el split (característica, umbral) que maximiza la ganancia de información.

        Evalúa, por fuerza bruta, todos los valores únicos de cada
        característica candidata como posible umbral.

        Args:
            X (np.ndarray): Características del nodo actual.
            y (np.ndarray): Etiquetas del nodo actual.
            feat_idxs (np.ndarray): Índices de características candidatas
                a evaluar (subconjunto aleatorio de columnas).

        Returns:
            tuple[int | None, float | None]: Índice de la mejor
            característica y su umbral óptimo. ``(None, None)`` si ningún
            split produce ganancia positiva.
        """
        best_gain = -np.inf
        best_feat, best_thresh = None, None

        for feat_idx in feat_idxs:
            col = X[:, feat_idx]
            thresholds = np.unique(col)

            for thresh in thresholds:
                gain = self._info_gain(y, col, thresh)

                if gain > best_gain:
                    best_gain = gain
                    best_feat = feat_idx
                    best_thresh = thresh

        return best_feat, best_thresh

    def _info_gain(self, y, col, thresh):
        """Calcula la ganancia de información de dividir en ``thresh``.

        Ganancia = entropía del padre - entropía ponderada de los hijos.

        Args:
            y (np.ndarray): Etiquetas del nodo actual.
            col (np.ndarray): Valores de la característica evaluada.
            thresh (float): Umbral candidato (izquierda si ``col <= thresh``).

        Returns:
            float: Ganancia de información. 0 si el split deja un lado vacío.
        """
        left_mask = col <= thresh
        right_mask = ~left_mask

        if not np.any(left_mask) or not np.any(right_mask):
            return 0

        parent_entropy = self._entropy(y)

        n = len(y)
        n_left = np.sum(left_mask)
        n_right = np.sum(right_mask)

        left_entropy = self._entropy(y[left_mask])
        right_entropy = self._entropy(y[right_mask])

        child_entropy = (n_left / n) * left_entropy + (n_right / n) * right_entropy

        return parent_entropy - child_entropy

    def _entropy(self, y):
        """Calcula la entropía de Shannon de un conjunto de etiquetas.

        Args:
            y (np.ndarray): Vector de etiquetas.

        Returns:
            float: Entropía en bits (0 = nodo puro, mayor = más mezcla de clases).
        """
        _, counts = np.unique(y, return_counts=True)
        probs = counts / len(y)
        # Se suma 1e-10 dentro del log para evitar log2(0) cuando alguna
        # probabilidad es 0 (no debería ocurrir dado que counts > 0, pero
        # protege ante casos límite numéricos).
        return -np.sum(probs * np.log2(probs + 1e-10))

    def _most_common(self, y):
        """Determina la clase mayoritaria de un conjunto de etiquetas.

        Args:
            y (np.ndarray): Vector de etiquetas.

        Returns:
            La clase más frecuente en ``y``.
        """
        return Counter(y).most_common(1)[0][0]

    def predict(self, X):
        """Predice la clase de cada muestra combinando el voto de todos los árboles.

        Args:
            X (np.ndarray): Matriz de características, shape (n_samples, n_features).

        Returns:
            np.ndarray: Predicciones de shape (n_samples,), obtenidas por
            votación mayoritaria (promedio redondeado) de los árboles del
            bosque.
        """
        print(f"Prediciendo sobre {X.shape[0]} muestras...")
        predictions = np.array([self._predict_tree(X, tree) for tree in self.trees])
        print("Predicción completa")
        return np.round(np.mean(predictions, axis=0))

    def _predict_tree(self, X, node):
        """Recorre un árbol individual y predice para cada fila de ``X``.

        Args:
            X (np.ndarray): Características a predecir.
            node (Node): Nodo (raíz o subárbol) desde el que se predice.

        Returns:
            np.ndarray: Predicciones de shape (X.shape[0],) según ese árbol.
        """
        if node.value is not None:
            return np.full(X.shape[0], node.value)

        col = X[:, node.feature_idx]
        left_mask = col <= node.threshold
        right_mask = ~left_mask

        predictions = np.zeros(X.shape[0])

        if np.any(left_mask):
            predictions[left_mask] = self._predict_tree(X[left_mask], node.left)
        if np.any(right_mask):
            predictions[right_mask] = self._predict_tree(X[right_mask], node.right)

        return predictions


# Ejemplo de uso: entrena y evalúa el Random Forest sobre un dataset
# Uso de dataset load_breast_cancer gratuito para probar, a través de sklearn.datasets
if __name__ == "__main__":
    X, y = load_breast_cancer(return_X_y=True)

    # Mezclar antes de partir: el dataset no viene aleatorizado, así que
    # sin esto el split 80/20 dependería del orden original de las filas.
    rng = np.random.default_rng(42)
    perm = rng.permutation(len(X))
    X, y = X[perm], y[perm]

    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    rf = RandomForest(n_trees=3, max_depth=4, min_samples_split=20)
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    accuracy = np.mean(y_pred == y_test)

    baseline = max(np.mean(y_test == 0), np.mean(y_test == 1))
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="binary")

    print(f"Accuracy:  {accuracy:.4f}  (baseline clase mayoritaria: {baseline:.4f})")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print("\nMatriz de confusión:")
    print(confusion_matrix(y_test, y_pred))
    print("\nReporte completo:")
    print(classification_report(y_test, y_pred))
