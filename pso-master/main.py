import threading
from numpy import exp, sqrt, cos, e, pi
import numpy as np
import matplotlib.pyplot as plt
from pso import PSO, ackley, griewank

MAX_ITER = 50000
BOUNDS = (-32, 32)
V_FACT = 1
DIM = 5
C1 = C2 = 2.05
INERTIA = 1
CONSTRICT_FACT = 1

# Função para executar o PSO para uma função objetivo específica e gerar o gráfico
def run_pso(objective_func, history_list, best_ever_list, subplot_idx, axs):
    history, best_ever = PSO(
        max_iter=MAX_ITER,
        bounds=BOUNDS,
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
thread_ackley = threading.Thread(target=run_pso, args=(ackley, history_list, best_ever_list, 0, axs))
thread_griewank = threading.Thread(target=run_pso, args=(griewank, history_list, best_ever_list, 1, axs))

# Iniciar as threads
thread_ackley.start()
thread_griewank.start()

# Esperar as threads terminarem
thread_ackley.join()
thread_griewank.join()

# Gerar o gráfico lado a lado apenas após as threads terem terminado
plt.tight_layout()

# Salvar a imagem
plt.savefig(f'pso_convergence.png')
plt.show()

