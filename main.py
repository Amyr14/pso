from numpy import exp, sqrt, cos, e, pi
import numpy as np
from pso import PSO, ackley
import matplotlib.pyplot as plt

MAX_ITER = 50000
BOUNDS = (-32, 32)
V_FACT = 1
DIM=2

history, best_ever = PSO(
    max_iter=MAX_ITER,
    bounds=BOUNDS,
    objective=ackley,
    dim=DIM,
    cog_fact=1.05,
    social_fact=2.05,
)

print('=========== melhor solução ===========')
print(best_ever)
plt.plot(history)
plt.savefig('pso_ackley.png')