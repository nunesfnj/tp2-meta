import numpy as np

def mutacao_crossover(populacao, idx_atual, bounds, F=0.8, Cr=0.7):
    # Seleção de 3 vetores aleatórios distintos
    indices = [i for i in range(len(populacao)) if i != idx_atual]
    a, b, c = populacao[np.random.choice(indices, 3, replace=False)]
    
    # Mutação: V = A + F * (B - C)
    doador = a + F * (b - c)
    
    # Crossover Binomial
    ponto_corte = np.random.rand(len(doador)) < Cr
    filho = np.where(ponto_corte, doador, populacao[idx_atual])
    
    # Reentry: Garante que o filho esteja dentro dos limites (Bounds)
    for j in range(len(bounds)):
        if filho[j] < bounds[j][0] or filho[j] > bounds[j][1]:
            filho[j] = np.random.uniform(bounds[j][0], bounds[j][1])
            
    return filho

def evolucao_diferencial(pop_size, generations, bounds, funcao_objetivo, funcao_comparacao):
    dim = len(bounds)
    # Inicialização da população [cite: 11]
    pop = np.random.uniform([b[0] for b in bounds], [b[1] for b in bounds], (pop_size, dim))
    
    # Avaliação inicial
    fitness_pop = np.array([funcao_objetivo(ind) for ind in pop])
    
    for gen in range(generations):
        for i in range(pop_size):
            filho = mutacao_crossover(pop, i, bounds)
            
            # A decisão de quem vence depende do critério de restrição [cite: 51]
            if funcao_comparacao(filho, pop[i]):
                pop[i] = filho
    
    # Retorna o melhor indivíduo da última geração
    melhores_fitness = [funcao_objetivo(ind) for ind in pop]
    idx_melhor = np.argmin(melhores_fitness)
    
    return pop[idx_melhor], melhores_fitness[idx_melhor]