import numpy as np

def objetivo_p1(vars):
    x1, x2, u1, u2 = vars
    return x1**0.6 + x2**0.6 - 6*x1 - 4*u1 + 3*u2 [cite: 19]

def violacao_p1(vars):
    x1, x2, u1, u2 = vars
    eps = 0.0001 # 
    
    # Restrições transformadas em g(x) <= 0 [cite: 13, 14, 15]
    h1 = abs(x2 - 3*x1 - 3*u1) - eps # Igualdade [cite: 20]
    g1 = x1 + 2*u1 - 4 # Desigualdade 1 [cite: 21]
    g2 = x2 + 2*u2 - 4 # Desigualdade 2 [cite: 22]
    
    # Soma apenas as violações (valores > 0)
    total_violacao = max(0, h1) + max(0, g1) + max(0, g2)
    return total_violacao

# O método E-constrained compara dois indivíduos:
# 1. Se ambos são viáveis (violação < epsilon_gen), vence o de menor f(x)
# 2. Se um é viável e o outro não, vence o viável
# 3. Se ambos são inviáveis, vence o de menor violação total