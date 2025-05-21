import os
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import json
import pandas as pd
from pso import PSO, ackley, griewank  # Importando a implementação do PSO

# Configurações de diretório
RESULTS_DIR = './results/pso_boxplots_inertia'
if not os.path.exists(RESULTS_DIR):
    os.mkdir(RESULTS_DIR)

# Definir valores de C1 e C2 para experimentos
C_VALUES = 1.55
NUM_EXECUTIONS = 10
MAX_ITER = 2000
BOUNDS = (-32, 32)
DIM = 10
V_FACT = 1
INERTIA = [0.1,0.5,0.7]
CONSTRICT_FACT = 1

# Função para rodar o PSO e coletar os melhores resultados
def run_pso_experiment(objective_func, inertia, results_list):
    best_ever_values = []
    for _ in range(NUM_EXECUTIONS):
        history, best_ever = PSO(
            bounds=BOUNDS,
            objective=objective_func,
            dim=DIM,
            max_iter=MAX_ITER,
            inertia=inertia,
            constrict_fact=CONSTRICT_FACT,
            v_fact=V_FACT,
            cog_fact=C_VALUES,
            social_fact=C_VALUES  # Como C1 = C2
        )
        best_ever_values.append(best_ever)
    results_list.extend(best_ever_values)

# Preparar a estrutura para armazenar os resultados dos experimentos
results_ackley = {i: [] for i in INERTIA}
results_griewank = {i: [] for i in INERTIA}

# Rodar os experimentos para as duas funções objetivo (Ackley e Griewank)
for i in INERTIA:
    # Rodar para Ackley
    run_pso_experiment(ackley, i, results_ackley[i])
    # Rodar para Griewank
    run_pso_experiment(griewank, i, results_griewank[i])

# Criar os DataFrames
ackley_df = pd.DataFrame({f'W={i}': results_ackley[i] for i in INERTIA})
griewank_df = pd.DataFrame({f'W={i}': results_griewank[i] for i in INERTIA})

# Calcular média e desvio padrão para Ackley
ackley_mean_std = {
    i: {
        'mean': np.mean(results_ackley[i]),
        'std': np.std(results_ackley[i])
    }
    for i in INERTIA
}

# Calcular média e desvio padrão para Griewank
griewank_mean_std = {
    i: {
        'mean': np.mean(results_griewank[i]),
        'std': np.std(results_griewank[i])
    }
    for i in INERTIA
}

# Organizar as médias e desvios padrão em um único dicionário
mean_std_dict = {
    'Ackley': ackley_mean_std,
    'Griewank': griewank_mean_std
}

# Salvar o dicionário em um arquivo JSON
path = "mean_std_inertia_" + str(MAX_ITER) + "_iter_"+ str(DIM) + "DIM" + ".json"
with open(os.path.join(RESULTS_DIR, path), mode='w') as fp:
    mean_std_json = json.dumps(mean_std_dict, indent=4, ensure_ascii=False)
    fp.write(mean_std_json)

# Criar a figura
fig, axes = plt.subplots(1, 2, figsize=(14, 7))

# Boxplot para Ackley
ackley_df.plot(kind='box', ax=axes[0])
axes[0].set_title('PSO - Ackley')
axes[0].set_ylabel('Melhor valor (Best Ever)')
axes[0].set_xlabel('Inertia')
axes[0].tick_params(axis='y')

# Boxplot para Griewank
griewank_df.plot(kind='box', ax=axes[1])
axes[1].set_title('PSO - Griewank')
axes[1].set_ylabel('Melhor valor (Best Ever)')
axes[1].set_xlabel('Inertia')
axes[1].tick_params(axis='y')

# Ajustar o layout
plt.tight_layout()

# Salvar o gráfico
path = "pso_boxplots" + str(MAX_ITER) + "_iter_"+ str(DIM) + "DIM" + ".png"
plt.savefig(os.path.join(RESULTS_DIR, path))
plt.show()
