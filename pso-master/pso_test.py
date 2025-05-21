import numpy as np
import matplotlib.pyplot as plt

C1 = C2 = 1.05
DIM = 2
MAX_ITER = 10000

def griewank(sol):
    sol = np.array(sol)
    DIM = len(sol)
    top1 = np.sum(sol ** 2)
    top2 = np.prod(np.cos(sol / np.sqrt(np.arange(1, DIM + 1))))  # Versão sem conversão para graus
    top = (1.0 / 4000.0) * top1 - top2 + 1
    return top
    #return 0.0 if abs(top) < 1e-10 else top

def ackley(sol):
    DIM = len(sol)
    sol = np.array(sol)
    aux = np.sum(sol ** 2)
    aux1 = np.sum(np.cos(2.0 * np.pi * sol))
    term1 = -20.0 * np.exp(-0.2 * np.sqrt(aux / DIM))
    term2 = -np.exp(aux1 / DIM)
    result = term1 + term2 + 20.0 + np.e
    #return result
    return 0.0 if abs(result) < 1e-10 else result

# === PSO ===
def pso(objective_func, num_particles=30, bounds=(-32, 32)):
    w = 0.7  # Inércia

    lower_bound, upper_bound = bounds

    # Inicialização das partículas
    particles = np.random.uniform(lower_bound, upper_bound, (num_particles, DIM))
    velocities = np.random.uniform(-1, 1, (num_particles, DIM))

    # Avaliar função objetivo
    fitness = np.array([objective_func(p) for p in particles])

    # Melhor posição pessoal
    pbest = particles.copy()
    pbest_val = fitness.copy()

    # Melhor posição global
    gbest_idx = np.argmin(fitness)
    gbest = particles[gbest_idx].copy()
    gbest_val = fitness[gbest_idx]

    # Lista para armazenar o melhor valor encontrado em cada iteração
    convergence = []

    # Loop principal
    for iteration in range(MAX_ITER):
        for i in range(num_particles):
            r1 = np.random.rand(DIM)
            r2 = np.random.rand(DIM)

            # Atualiza velocidade
            velocities[i] = (w * velocities[i] +
                             C1 * r1 * (pbest[i] - particles[i]) +
                             C2 * r2 * (gbest - particles[i]))

            # Atualiza posição
            particles[i] += velocities[i]

            # Respeita os limites
            particles[i] = np.clip(particles[i], lower_bound, upper_bound)

            # Avaliar nova posição
            current_val = objective_func(particles[i])
            if current_val < pbest_val[i]:
                pbest[i] = particles[i]
                pbest_val[i] = current_val

                if current_val < gbest_val:
                    gbest = particles[i]
                    gbest_val = current_val

        # Armazena o melhor valor encontrado até o momento
        convergence.append(gbest_val)

        # Imprime a iteração e o melhor valor encontrado a cada 100 iterações (opcional)
        if (iteration + 1) % 100 == 0:
            print(f"Iteração {iteration + 1}/{MAX_ITER}, Melhor valor: {gbest_val:.6f}")

    return gbest, gbest_val, convergence

# Otimizando a função Griewank
best_pos, best_val, convergence = pso(ackley)

print("\nMelhor solução encontrada (Griewank):", best_pos)
print("Valor mínimo encontrado:", best_val)

# Plotando o gráfico de convergência com escala logarítmica no eixo Y
plt.plot(convergence)
plt.xlabel('Número de iterações')
plt.ylabel('Melhor valor encontrado até o momento (Escala logarítmica)')
plt.title('Gráfico de Convergência do PSO')
plt.grid(True)

# Usando escala logarítmica no eixo Y para visualização melhorada
plt.yscale('log')

plt.show()
