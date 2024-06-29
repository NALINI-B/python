import random
import math
import matplotlib.pyplot as plt

class LineModel:
    def __init__(self, points):
        self.points = points
        self.a = 1  # Default slope
        self.b = -1  # Default b value
        self.c = 0  # Default intercept

    def plot(self):
        X, Y = zip(*self.points)
        X_min, X_max = min(X), max(X)
        
        # Calculate the fitted line for plotting
        X_line = [X_min + i * (X_max - X_min) / 100 for i in range(101)]
        Y_line = [-(self.a * x + self.c) / self.b for x in X_line]
        
        # Plot the results
        plt.scatter(X, Y, label='Data Points')
        plt.plot(X_line, Y_line, color='red', label='Fitted Line')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.legend()
        plt.show()

class OptimizeLineModel(LineModel):
    def __init__(self, points):
        super().__init__(points)

    def calculate_distance(self, a, b, c, x, y):
        return abs(a * x + b * y + c) / math.sqrt(a**2 + b**2)

    def objective(self, a, b, c):
        total_distance = 0
        for x, y in self.points:
            total_distance += self.calculate_distance(a, b, c, x, y)
        return total_distance

    def optimize(self, learning_rate=0.01, iterations=1000):
        a, b, c = self.a, self.b, self.c
        for _ in range(iterations):
            grad_a, grad_b, grad_c = 0, 0, 0
            for x, y in self.points:
                distance = self.calculate_distance(a, b, c, x, y)
                sign = 1 if a * x + b * y + c >= 0 else -1
                common_term = sign / math.sqrt(a**2 + b**2)
                grad_a += common_term * x
                grad_b += common_term * y
                grad_c += common_term
            a -= learning_rate * grad_a
            b -= learning_rate * grad_b
            c -= learning_rate * grad_c

        self.a, self.b, self.c = a, b, c
        print(f'Optimized parameters: a = {self.a}, b = {self.b}, c = {self.c}')

# Example usage
points = [(0.1, 0.2), (0.4, 0.5), (0.6, 0.7), (0.9, 1.0)]
line_model = LineModel(points)
line_model.plot()

opt_line_model = OptimizeLineModel(points)
opt_line_model.optimize()
opt_line_model.plot()
