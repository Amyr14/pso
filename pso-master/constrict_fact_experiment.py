import threading
from numpy import exp, sqrt, cos, e, pi
import numpy as np
import matplotlib.pyplot as plt
from pso import PSO, ackley, griewank
import os

RESULTS_DIR = './results/convergence_constrict_fact'
if not os.path.exists(RESULTS_DIR):
    os.mkdir(RESULTS_DIR)

MAX_ITER = 2000
V_FACT = 1
DIM = 5
C1 = C2 = 2.05
INERTIA = 0.1
CONSTRICT_FACT = (2 / abs(2 - (C1 + C2) - sqrt(pow(C1 + C2, 2) - 4 * (C1 + C2))))

# Função para executar o PSO para uma função objetivo específica e gerar o gráfico
def run_pso(objective_func, history_list, best_ever_list, subplot_idx, axs,bounds_func):
    history, best_ever = PSO(
        max_iter=MAX_ITER,
        bounds=bounds_func,
        objective=objective_func,
        dim=DIM,
        cog_fact=C1,
        social_fact=C2,
        v_fact=V_FACT,
        inertia=INERTIA,
        constrict_fact=CONSTRICT_FACT
    )
    
    history_list[subplot_idx] = history
    best_ever_list[subplot_idx] = best_ever
    
    # Gerar o gráfico (essa parte agora é mantida no subplot já correspondente)
    ax = axs[subplot_idx]
    ax.plot(history)
    ax.set_xlabel('Iterações')
    ax.set_ylabel('Best Ever')
    ax.set_title(f'Convergência - {objective_func.__name__} - Dimensões: {DIM}')
    ax.grid(True)
    ax.set_yscale('log')

# Preparar os dados para as duas funções objetivo
history_list = [None, None]
best_ever_list = [None, None]

# Criar os subgráficos
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

# Criar as threads para executar PSO para as duas funções objetivo
thread_ackley = threading.Thread(target=run_pso, args=(ackley, history_list, best_ever_list, 0, axs,(-32,32)))
thread_griewank = threading.Thread(target=run_pso, args=(griewank, history_list, best_ever_list, 1, axs,(-600,600)))

# Iniciar as threads
thread_ackley.start()
thread_griewank.start()

# Esperar as threads terminarem
thread_ackley.join()
thread_griewank.join()

# Gerar o gráfico lado a lado apenas após as threads terem terminado
plt.tight_layout()

# Salvar a imagem
path = "pso_convergence_cfact_" + str(MAX_ITER) + "Iter_" + str(DIM) + "DIM.png"
plt.savefig(os.path.join(RESULTS_DIR, path))
plt.show()

