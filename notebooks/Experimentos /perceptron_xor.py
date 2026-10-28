import numpy as np
import matplotlib.pyplot as plt

# Un perceptrón de una sola capa NO puede resolver XOR (una sola recta no separa las clases).
# Con DOS rectas (dos neuronas ocultas) + una neurona de salida, sí es posible.

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


class PerceptronXOR:
    def __init__(self, learning_rate=2.0, epochs=16, steps_per_epoch=50, seed=0):
        self.lr = learning_rate
        self.epochs = epochs
        # Cada "época" = steps_per_epoch pasos de descenso de gradiente
        # (con solo 1 paso por época, 16 épocas no alcanzan para resolver XOR).
        self.steps_per_epoch = steps_per_epoch
        rng = np.random.default_rng(seed)
        # Capa oculta: 2 neuronas = 2 rectas en el plano
        self.W1 = rng.normal(0, 1.0, (2, 2))
        self.b1 = np.zeros(2)
        # Capa de salida: combina las dos rectas
        self.W2 = rng.normal(0, 1.0, (2, 1))
        self.b2 = np.zeros(1)

    def forward(self, X):
        self.h = sigmoid(X @ self.W1 + self.b1)
        self.out = sigmoid(self.h @ self.W2 + self.b2)
        return self.out

    def train_step(self, X, y):
        y = y.reshape(-1, 1)
        out = self.forward(X)
        d_out = (out - y) * out * (1 - out)
        d_h = (d_out @ self.W2.T) * self.h * (1 - self.h)
        self.W2 -= self.lr * self.h.T @ d_out
        self.b2 -= self.lr * d_out.sum(axis=0)
        self.W1 -= self.lr * X.T @ d_h
        self.b1 -= self.lr * d_h.sum(axis=0)
        return float(np.mean((out - y) ** 2))

    def train_and_plot(self, X, y, title):
        # Entrenar las 16 épocas (sin graficar las intermedias)
        loss = None
        for epoch in range(1, self.epochs + 1):
            for _ in range(self.steps_per_epoch):
                loss = self.train_step(X, y)

        preds = (self.forward(X).ravel() >= 0.5).astype(int)
        print(f"[{title}] Predicciones finales: {preds.tolist()}  (esperado: {y.astype(int).tolist()})")

        # Graficar SOLO la última época
        fig, ax = plt.subplots(figsize=(7, 6))

        # Mapa de calor de la salida de la red sobre el plano
        xx, yy = np.meshgrid(np.linspace(-0.5, 1.5, 300), np.linspace(-0.5, 1.5, 300))
        grid = np.c_[xx.ravel(), yy.ravel()]
        zz = self.forward(grid).reshape(xx.shape)
        im = ax.imshow(zz, extent=(-0.5, 1.5, -0.5, 1.5), origin="lower",
                       cmap="coolwarm", vmin=0, vmax=1, alpha=0.9)
        fig.colorbar(im, ax=ax, label="Salida de la red")

        # Las dos rectas (una por neurona oculta): w1*x1 + w2*x2 + b = 0
        x_vals = np.array([-0.5, 1.5])
        colors = ["black", "lime"]
        for k in range(2):
            w1, w2 = self.W1[0, k], self.W1[1, k]
            b = self.b1[k]
            label = f"Recta {k + 1} (Época {self.epochs})"
            if abs(w2) > 1e-9:
                y_vals = (-w1 * x_vals - b) / w2
                ax.plot(x_vals, y_vals, "--", color=colors[k], linewidth=2, label=label)
            else:
                ax.axvline(-b / w1, linestyle="--", color=colors[k], linewidth=2, label=label)

        # Puntos de datos
        for xi, t in zip(X, y):
            ax.scatter(xi[0], xi[1], s=150, marker="o" if t == 1 else "X",
                       c="white", edgecolors="black", linewidths=2, zorder=5)
            ax.text(xi[0] + 0.06, xi[1] + 0.06, f"({int(xi[0])},{int(xi[1])})={int(t)}",
                    fontsize=10, zorder=6,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7))

        ax.set_xlim(-0.5, 1.5)
        ax.set_ylim(-0.5, 1.5)
        ax.axhline(0, color="grey", linewidth=1)
        ax.axvline(0, color="grey", linewidth=1)
        ax.grid(True, linestyle=":", alpha=0.7)
        ax.set_title(f"{title} - Época {self.epochs}\n"
                     f"Error={loss:.4f}  (O = 1, X = 0)")
        ax.set_xlabel("X1")
        ax.set_ylabel("X2")
        ax.legend(loc="lower right")
        fig.tight_layout()
        fig.savefig("xor_heatmap.png", dpi=130)
        plt.show()


# --- DATOS ---
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
y_xor = np.array([0, 1, 1, 0], dtype=float)

net = PerceptronXOR(learning_rate=2.0, epochs=16, steps_per_epoch=50, seed=0)
print("Ejecutando visualización para COMPUERTA XOR...")
net.train_and_plot(X, y_xor, "Compuerta XOR (No Linealmente Separable, 2 rectas)")