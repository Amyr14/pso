import os
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import json
import pandas as pd
from pso import PSO, ackley, griewank  # Importando a implementação do PSO

# Configurações de diretório
RESULTS_DIR = './results/pso_boxplots'
if not os.path.exists(RESULTS_DIR):
    os.mkdir(RESULTS_DIR)

# Definir valores de C1 e C2 para experimentos
C_VALUES = [1.55, 2.05, 3.55]
NUM_EXECUTIONS = 10
MAX_ITER = 2000
BOUNDS = (-32, 32)
DIM = 10
V_FACT = 1
INERTIA = 1
CONSTRICT_FACT = 1

# Função para rodar o PSO e coletar os melhores resultados
def run_pso_experiment(objective_func, c, results_list):
    best_ever_values = []
    for _ in range(NUM_EXECUTIONS):
        history, best_ever = PSO(
            bounds=BOUNDS,
            objective=objective_func,
            dim=DIM,
            max_iter=MAX_ITER,
            inertia=INERTIA,
            constrict_fact=CONSTRICT_FACT,
            v_fact=V_FACT,
            cog_fact=c,
            social_fact=c  # Como C1 = C2
        )
        best_ever_values.append(best_ever)
    results_list.extend(best_ever_values)

# Preparar a estrutura para armazenar os resultados dos experimentos
results_ackley = {c: [] for c in C_VALUES}
results_griewank = {c: [] for c in C_VALUES}

# Rodar os experimentos para as duas funções objetivo (Ackley e Griewank)
for c in C_VALUES:
    # Rodar para Ackley
    run_pso_experiment(ackley, c, results_ackley[c])
    # Rodar para Griewank
    run_pso_experiment(griewank, c, results_griewank[c])

# Criar os DataFrames
ackley_df = pd.DataFrame({f'C={c}': results_ackley[c] for c in C_VALUES})
griewank_df = pd.DataFrame({f'C={c}': results_griewank[c] for c in C_VALUES})

# Calcular média e desvio padrão para Ackley
ackley_mean_std = {
    c: {
        'mean': np.mean(results_ackley[c]),
        'std': np.std(results_ackley[c])
    }
    for c in C_VALUES
}

# Calcular média e desvio padrão para Griewank
griewank_mean_std = {
    c: {
        'mean': np.mean(results_griewank[c]),
        'std': np.std(results_griewank[c])
    }
    for c in C_VALUES
}

# Organizar as médias e desvios padrão em um único dicionário
mean_std_dict = {
    'Ackley': ackley_mean_std,
    'Griewank': griewank_mean_std
}

# Salvar o dicionário em um arquivo JSON
with open(os.path.join(RESULTS_DIR, 'mean_std.json'), mode='w') as fp:
    mean_std_json = json.dumps(mean_std_dict, indent=4, ensure_ascii=False)
    fp.write(mean_std_json)

# Criar a figura
fig, axes = plt.subplots(1, 2, figsize=(14, 7))

# Boxplot para Ackley
ackley_df.plot(kind='box', ax=axes[0])
axes[0].set_title('PSO - Ackley')
axes[0].set_ylabel('Melhor valor (Best Ever)')
axes[0].set_xlabel('C1 = C2')
axes[0].tick_params(axis='y')

# Boxplot para Griewank
griewank_df.plot(kind='box', ax=axes[1])
axes[1].set_title('PSO - Griewank')
axes[1].set_ylabel('Melhor valor (Best Ever)')
axes[1].set_xlabel('C1 = C2')
axes[1].tick_params(axis='y')

# Ajustar o layout
plt.tight_layout()

# Salvar o gráfico
plt.savefig(os.path.join(RESULTS_DIR, 'pso_boxplots.png'))
plt.show()
