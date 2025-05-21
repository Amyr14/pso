import numpy as np
from numpy import pi

# Suportar n dimensõe
def ackley(x):
    x = np.atleast_1d(x)
    a = 20
    b = 0.2
    c = 2*pi

    if x.ndim == 1:
        sum_sq = np.sum(x**2)
        sum_cos = np.sum(np.cos(c * x))
    else:
        sum_sq = np.sum(x**2, axis=1)
        sum_cos = np.sum(np.cos(c * x), axis=1)
    
    n = x.shape[-1]  # Dimension of the input space
    
    term1 = -a * np.exp(-b * np.sqrt(sum_sq / n))
    term2 = -np.exp(sum_cos / n)
    result = term1 + term2 + a + np.exp(1)
    return result if result > 1e-10 else 0

def griewank(x):
    sol = np.array(x)  
    top1 = np.sum(sol ** 2)  
    top2 = np.prod(np.cos(sol / np.sqrt(np.arange(1, len(sol) + 1))))  
    top = (1.0 / 4000.0) * top1 - top2 + 1
    return top if top > 1e-10 else 0  # Retornar 0 se o valor for muito pequeno
    


class Particle:
    def __init__(self, dim, v_fact, bounds):
        self.dim = dim
        x_min, x_max = bounds
        v_max = v_fact * (x_max - x_min)
        self.pos = (x_max - x_min) * np.random.random_sample(dim) + x_min
        self.v = np.clip((x_max - x_min) * np.random.random_sample(dim) + x_min, -v_max, v_max)
        self.best_know = self.pos

def calculate_velocity(p, g_best, inertia, constrict_fact, v_fact, cog_fact, social_fact, bounds):
    x_min, x_max = bounds
    v_max = v_fact * (x_max - x_min)
    dim = p.pos.size
    rp = np.random.random_sample(dim)
    rg = np.random.random_sample(dim)
    v = constrict_fact * (inertia*p.v + cog_fact*rp*(p.best_know - p.pos) + social_fact*rg*(g_best - p.pos))
    return np.clip(v, -v_max, v_max)

# Parametrizar o fator de constrição
def PSO(bounds, objective, dim, max_iter=5000, num_p=30, inertia=1, constrict_fact=1, v_fact=1, cog_fact=1, social_fact=1):
    
    if bounds[0] >= bounds[1]:
        raise Exception('Restrição de domínio inválida') 
    
    particles = [Particle(dim, v_fact, bounds) for _ in range(num_p)]

    # Inicialização
    history = []
    x_min, x_max = bounds
    init_costs = list(map(lambda p: objective(p.pos), particles))
    g_best_i = np.argmin(init_costs)
    g_best = particles[g_best_i].pos
    g_best_cost = init_costs[g_best_i]
    i = 0
    
    while i < max_iter:
        i += 1
        #print(i)
        if (i + 1) % 100 == 0:
            print(f"Iteração {i + 1}/{max_iter}, Melhor valor: {g_best_cost:.6f} - [{objective.__name__}]")
        history.append(g_best_cost)

        if g_best_cost < 1e-10:
            print(f"Convergência atingida na iteração {i}.")
            break

        for p in particles:
            v = calculate_velocity(p, g_best, inertia, constrict_fact, v_fact, cog_fact, social_fact, bounds)
            p.pos = np.clip(p.pos + v, x_min, x_max)
            new_cost = objective(p.pos)
            
            if new_cost <= objective(p.best_know):
                p.best_know = p.pos
                if new_cost <= g_best_cost:
                    g_best = p.pos
                    g_best_cost = new_cost
        
    return history, g_best_cost