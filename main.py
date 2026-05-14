import numpy as np
import matplotlib.pyplot as plt
from tp2.ED import evolucao_diferencial

# --- CONFIGURAÇÕES DO PROBLEMA 1 --- [cite: 18, 23, 24, 25]
BOUNDS_P1 = [(0, 3), (0, 4), (0, 2), (0, 1)] # x1, x2, u1, u2 (limites estimados)
EPSILON_IGUALDADE = 0.0001 # [cite: 16]

def objetivo_p1(vars):
    x1, x2, u1, u2 = vars
    # Função a minimizar [cite: 19]
    return (x1**0.6) + (x2**0.6) - (6 * x1) - (4 * u1) + (3 * u2)

def calcular_violacao_p1(vars):
    x1, x2, u1, u2 = vars
    # Restrições g(x) <= 0 [cite: 20, 21, 22]
    h1 = abs(x2 - 3*x1 - 3*u1) - EPSILON_IGUALDADE 
    g1 = x1 + 2*u1 - 4
    g2 = x2 + 2*u2 - 4
    
    # Soma das violações positivas
    return sum(max(0, v) for v in [h1, g1, g2])

# --- CRITÉRIOS DE SELEÇÃO (TRATAMENTO DE RESTRIÇÕES) --- [cite: 52]

def criterio_penalidade_estatica(filho, pai):
    W = 1000 # Peso da penalidade
    
    f_filho = objetivo_p1(filho) + W * (calcular_violacao_p1(filho)**2)
    f_pai = objetivo_p1(pai) + W * (calcular_violacao_p1(pai)**2)
    
    return f_filho < f_pai

# --- EXECUÇÃO --- [cite: 54]

def rodar_experimento(nome, criterio):
    resultados = []
    print(f"Executando {nome}...")
    for i in range(30):
        _, melhor_val = evolucao_diferencial(
            pop_size=30, 
            generations=100, 
            bounds=BOUNDS_P1, 
            funcao_objetivo=objetivo_p1,
            funcao_comparacao=criterio
        )
        resultados.append(melhor_val)
    return resultados

if __name__ == "__main__":
    # Exemplo com Configuração A (Penalidade Estática)
    res_a = rodar_experimento("Configuração A (Estática)", criterio_penalidade_estatica)
    
    # Estatísticas básicas [cite: 55]
    print(f"\nResultados Config A:")
    print(f"Média: {np.mean(res_a):.4f}")
    print(f"Desvio Padrão: {np.std(res_a):.4f}")
    print(f"Mínimo: {np.min(res_a):.4f}")
    
    # Gerar Boxplot [cite: 56]
    plt.boxplot(res_a)
    plt.title("Boxplot Problema 1 - Configuração A")
    plt.ylabel("Valor da Função Objetivo")
    plt.show()