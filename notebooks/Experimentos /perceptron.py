import numpy as np
import matplotlib.pyplot as plt

class PerceptronVisual:
    def __init__(self, learning_rate=0.1, epochs=10):
        self.lr = learning_rate
        self.epochs = epochs
        self.weights = np.zeros(2)
        self.bias = 0.0

    def activation(self, x):
        return 1 if x >= 0 else 0

    def predict(self, inputs):
        return self.activation(np.dot(inputs, self.weights) + self.bias)

    def train_and_plot(self, X, y, title):
        # Reiniciar pesos para cada compuerta
        self.weights = np.array([0.5, -0.5]) # Pesos iniciales arbitrarios para ver la recta cruzando el plano
        self.bias = -0.2
        
        plt.figure(figsize=(6, 6))
        
        for epoch in range(self.epochs):
            error_flag = False
            for xi, target in zip(X, y):
                pred = self.predict(xi)
                error = target - pred
                if error != 0:
                    self.weights += self.lr * error * xi
                    self.bias += self.lr * error
                    error_flag = True
            
            # Graficar el estado de la frontera de decisión en cada época
            plt.clf()
            
            # Dibujar los puntos de los datos
            for i, sample in enumerate(X):
                color = 'blue' if y[i] == 1 else 'red'
                marker = 'o' if y[i] == 1 else 'x'
                plt.scatter(sample[0], sample[1], color=color, s=150, marker=marker, linewidths=3)
                plt.text(sample[0] + 0.05, sample[1], f"({sample[0]},{sample[1]})={y[i]}", fontsize=10)

            # Ecuación de la línea: w1*x1 + w2*x2 + bias = 0  =>  x2 = (-w1*x1 - bias) / w2
            if self.weights[1] != 0:
                x_vals = np.array([-0.5, 1.5])
                y_vals = (-self.weights[0] * x_vals - self.bias) / self.weights[1]
                plt.plot(x_vals, y_vals, '--k', label=f'Época {epoch+1}')

            plt.xlim(-0.5, 1.5)
            plt.ylim(-0.5, 1.5)
            plt.axhline(0, color='grey', linewidth=1)
            plt.axvline(0, color='grey', linewidth=1)
            plt.grid(True, linestyle=':', alpha=0.7)
            plt.title(f"{title} - Época {epoch+1}\nPesos: W1={self.weights[0]:.2f}, W2={self.weights[1]:.2f}, b={self.bias:.2f}")
            plt.xlabel("X1")
            plt.ylabel("X2")
            plt.legend()
            
            plt.pause(0.8) # Pausa dramática para que los alumnos vean el movimiento en vivo
            
            if not error_flag:
                print(f"[{title}] ¡Convergencia alcanzada en la época {epoch+1}!")
                break
        
        plt.show()

# --- DATOS ---
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

# 1. Compuerta AND (Converge con éxito)
y_and = np.array([0, 0, 0, 1])
perceptron_and = PerceptronVisual(learning_rate=0.2, epochs=8)
print("Ejecutando visualización para COMPUERTA AND...")
perceptron_and.train_and_plot(X, y_and, "Compuerta AND (Linealmente Separable)")

# 2. Compuerta XOR (Falla / Oscila)
y_xor = np.array([0, 1, 1, 0])
perceptron_xor = PerceptronVisual(learning_rate=0.2, epochs=8)
print("\nEjecutando visualización para COMPUERTA XOR...")
perceptron_xor.train_and_plot(X, y_xor, "Compuerta XOR (No Linealmente Separable)")
