import pulp
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# Check available solvers
print("Available solvers:", pulp.listSolvers(onlyAvailable=True))

# Define the Linear Programming Problem
model = pulp.LpProblem("Maximize_Profit", pulp.LpMaximize)

# Define Decision Variables
x1 = pulp.LpVariable("Product_A", lowBound=0, cat='Continuous')
x2 = pulp.LpVariable("Product_B", lowBound=0, cat='Continuous')

# Define Objective Function (Maximize Profit)
model += 50 * x1 + 40 * x2, "Total Profit"

# Define Constraints
model += 5 * x1 + 3 * x2 <= 240, "Labor Constraint"
model += 2 * x1 + 1 * x2 <= 100, "Material Constraint"
model += 1 * x1 + 2 * x2 <= 80, "Machine Hours Constraint"

# Use COIN_CMD solver (since it's available on your system)
solver = pulp.COIN_CMD(msg=False)

# Solve the model
status = model.solve(solver)

# Check if the solver ran successfully
if status != pulp.LpStatusOptimal:
    raise Exception("Solver did not find an optimal solution. Try using a different solver.")

# Print the Results
print("Status:", pulp.LpStatus[model.status])
print(f"Optimal Production Plan: Product A: {x1.varValue}, Product B: {x2.varValue}")
print(f"Maximum Profit: ${pulp.value(model.objective)}")

# Visualization of Constraints
x = np.linspace(0, 60, 100)

def labor_constraint(x):
    return (240 - 5*x) / 3

def material_constraint(x):
    return (100 - 2*x)

def machine_constraint(x):
    return (80 - x) / 2

plt.figure(figsize=(8,6))
plt.plot(x, labor_constraint(x), label='Labor Constraint', linestyle='--', color='blue')
plt.plot(x, material_constraint(x), label='Material Constraint', linestyle='--', color='green')
plt.plot(x, machine_constraint(x), label='Machine Hours Constraint', linestyle='--', color='purple')

# Fill feasible region
y1 = labor_constraint(x)
y2 = material_constraint(x)
y3 = machine_constraint(x)
y_min = np.minimum(np.minimum(y1, y2), y3)
y_min[y_min < 0] = 0  # Ensure only positive values
plt.fill_between(x, 0, y_min, color='gray', alpha=0.3)

# Plot optimal solution
plt.scatter(x1.varValue, x2.varValue, color='red', zorder=5, label='Optimal Solution')

plt.xlabel("Product A")
plt.ylabel("Product B")
plt.legend()
plt.title("Feasible Region and Optimal Solution")
plt.grid()
plt.show()


# 3D Surface Plot: Profit as a function of Product A & B
X, Y = np.meshgrid(np.linspace(0, 60, 30), np.linspace(0, 60, 30))
Z = 50 * X + 40 * Y

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap="viridis", alpha=0.7)
ax.scatter(x1.varValue, x2.varValue, pulp.value(model.objective), color='red', s=100, label="Optimal Solution")

ax.set_xlabel("Product A")
ax.set_ylabel("Product B")
ax.set_zlabel("Profit")
ax.set_title("Profit Surface with Optimal Point")
plt.legend()
plt.show()

# Bar Chart: Optimal Production Plan
plt.figure(figsize=(6,5))
plt.bar(["Product A", "Product B"], [x1.varValue, x2.varValue], color=['blue', 'green'])
plt.xlabel("Products")
plt.ylabel("Quantity")
plt.title("Optimal Production Plan")
plt.show()
