"""
Exemplo matemático completo do módulo Fuzzy
Imagem: 4×4×3 (altura=4, largura=4, canais=3)

Vamos acompanhar UM pixel específico: posição (0,0)
através de TODAS as operações matemáticas.
"""

import numpy as np

np.random.seed(42)
np.set_printoptions(precision=4, suppress=True)

# ==============================================================================
# IMAGEM DE ENTRADA
# ==============================================================================

print("=" * 60)
print("IMAGEM DE ENTRADA: shape (1, 4, 4, 3)")
print("=" * 60)

# Simula saída de um bottleneck: valores entre -1 e 1 (após BatchNorm)
imagem = np.array([[
    # Linha 0
    [[ 0.8, -0.3,  0.5],  # pixel (0,0)
     [ 0.1,  0.9, -0.2],  # pixel (0,1)
     [-0.5,  0.4,  0.7],  # pixel (0,2)
     [ 0.3, -0.8,  0.1]], # pixel (0,3)
    # Linha 1
    [[ 0.6,  0.2, -0.4],
     [-0.1,  0.7,  0.3],
     [ 0.9, -0.5,  0.8],
     [-0.3,  0.1,  0.6]],
    # Linha 2
    [[ 0.4, -0.6,  0.2],
     [ 0.7,  0.3, -0.9],
     [-0.2,  0.8,  0.5],
     [ 0.5, -0.1,  0.4]],
    # Linha 3
    [[-0.7,  0.5,  0.3],
     [ 0.2, -0.4,  0.8],
     [ 0.6,  0.1, -0.3],
     [-0.4,  0.9,  0.2]],
]], dtype=np.float32)  # shape: (1, 4, 4, 3)

print(f"Shape: {imagem.shape}  →  (batch=1, H=4, W=4, C=3 canais)\n")
print("Pixel (0,0) — os 3 canais:")
print(f"  Canal 0 = {imagem[0,0,0,0]:.4f}")
print(f"  Canal 1 = {imagem[0,0,0,1]:.4f}")
print(f"  Canal 2 = {imagem[0,0,0,2]:.4f}")


# ==============================================================================
# PASSO 1: FUZZIFICAÇÃO
# ==============================================================================
# Método: Funções de pertinência GAUSSIANAS
#
# Fórmula:
#         μ(x) = exp( -(x - c)² / (2σ²) )
#
# Onde:
#   x = valor do pixel no canal k
#   c = centro do conjunto fuzzy (Baixo / Médio / Alto)
#   σ = largura (quanto a gaussiana é "aberta" ou "fechada")
#
# Resultado: para cada pixel e cada canal, temos 3 graus de pertinência.
# ==============================================================================

print("\n" + "=" * 60)
print("PASSO 1: FUZZIFICAÇÃO")
print("=" * 60)

# Parâmetros fuzzy (treináveis na rede; aqui fixamos para o exemplo)
# num_sets = 3  →  Baixo, Médio, Alto
# Para cada canal, temos 3 centros e 3 sigmas

#            Baixo   Médio   Alto
centros = np.array([
    [-0.8,   0.0,   0.8],   # canal 0
    [-0.8,   0.0,   0.8],   # canal 1
    [-0.8,   0.0,   0.8],   # canal 2
], dtype=np.float32)  # shape: (C=3, num_sets=3)

sigmas = np.array([
    [0.5,    0.5,   0.5],   # canal 0
    [0.5,    0.5,   0.5],   # canal 1
    [0.5,    0.5,   0.5],   # canal 2
], dtype=np.float32)  # shape: (C=3, num_sets=3)

nomes_conjuntos = ["Baixo", "Médio", "Alto"]

print("\nFórmula Gaussiana:")
print("  μ(x) = exp( -(x - c)² / (2σ²) )\n")

pixel_00 = imagem[0, 0, 0, :]  # shape: (3,)  →  valores dos 3 canais em (0,0)

print(f"Pixel (0,0) — valores brutos: {pixel_00}\n")

# Calcula pertinência para o pixel (0,0)
# x:       (3,)       →  expande para  (3, 1)
# centros: (3, 3)
# sigmas:  (3, 3)

x_exp = pixel_00[:, np.newaxis]           # (3, 1)
memberships_00 = np.exp(
    -((x_exp - centros) * 2) / (2 * sigmas * 2)
)  # shape: (3, 3)  →  (canal, conjunto_fuzzy)

print("Pertinências do pixel (0,0):\n")
print(f"{'':12} {'Baixo(c=-0.8)':>15} {'Médio(c=0.0)':>15} {'Alto(c=0.8)':>15}")
print("-" * 60)
for c in range(3):
    x_val = pixel_00[c]
    row = memberships_00[c]
    print(f"Canal {c} (x={x_val:+.2f}):  {row[0]:>15.4f} {row[1]:>15.4f} {row[2]:>15.4f}")

print("\nExplicação detalhada canal 0 (x=0.8):")
x0 = pixel_00[0]
for j, (c, s, nome) in enumerate(zip(centros[0], sigmas[0], nomes_conjuntos)):
    mu = np.exp(-((x0 - c)*2) / (2 * s*2))
    print(f"  μ_{nome}(0.8) = exp(-({x0:.1f} - {c:.1f})² / (2×{s:.1f}²))")
    print(f"           = exp(-({x0-c:.2f})² / {2*s**2:.2f})")
    print(f"           = exp(-{(x0-c)*2:.4f} / {2*s*2:.2f})")
    print(f"           = exp(-{(x0-c)*2/(2*s*2):.4f})")
    print(f"           = {mu:.4f}")

# Calcula para todos os pixels
# imagem: (1, 4, 4, 3)  →  expande para  (1, 4, 4, 3, 1)
x_all = imagem[..., np.newaxis]                     # (1, 4, 4, 3, 1)
memberships_all = np.exp(
    -((x_all - centros) * 2) / (2 * sigmas * 2)
)  # (1, 4, 4, 3, 3)

print(f"\nShape após fuzzificação: {memberships_all.shape}")
print("(batch=1, H=4, W=4, C=3 canais, num_sets=3 conjuntos)\n")


# ==============================================================================
# PASSO 2: INFERÊNCIA FUZZY (Takagi-Sugeno)
# ==============================================================================
# Método de Takagi-Sugeno (TSK):
#
# Em TSK, as PREMISSAS das regras são fuzzy (como em Mamdani),
# mas as CONSEQUÊNCIAS são funções lineares das entradas — não conjuntos fuzzy.
#
# Regra i:  SE x1 é A1i E x2 é A2i E ...  ENTÃO  y = w0 + w1*x1 + w2*x2 + ...
#
# Na nossa implementação:
#   - As premissas são os graus de pertinência (μ) de cada canal
#   - As consequências são APRENDIDAS pela rede (pesos W)
#   - Não usamos AND fuzzy explícito (t-norm), mas uma combinação linear
#     ponderada — isso é equivalente ao TSK de ordem 0 generalizado
#
# Implementação:
#   1. Achata memberships: (B, H, W, C*num_sets)  →  vetor de premissas
#   2. y_regra_r = Σ_k  W[k, r] * μ_k            →  combinação linear (TSK)
#   3. Aplica ReLU para manter ativações ≥ 0
# ==============================================================================

print("=" * 60)
print("PASSO 2: INFERÊNCIA FUZZY (Takagi-Sugeno)")
print("=" * 60)

print("""
Takagi-Sugeno (TSK):
  PREMISSA:    SE canal_k é Conjunto_j  (grau μ_kj)
  CONSEQUÊNCIA: função LINEAR dos graus de pertinência
  
  Ativação da regra r:
    y_r = ReLU( Σ_{k,j}  W[k*num_sets + j, r] × μ_kj )
    
  Onde W é a matriz de pesos aprendível (3×3=9 entradas → num_rules saídas)
""")

# Achata as pertinências: (1, 4, 4, 3, 3) → (1, 4, 4, 9)
B, H, W, C, NS = memberships_all.shape
flat = memberships_all.reshape(B, H, W, C * NS)   # (1, 4, 4, 9)

print(f"Shape após achatar: {flat.shape}")
print("(batch=1, H=4, W=4, C×num_sets = 3×3 = 9 valores por pixel)\n")

print("Vetor achatado do pixel (0,0):")
flat_00 = flat[0, 0, 0, :]   # (9,)
labels = [f"μ_c{c}_{nome[:3]}" for c in range(3) for nome in nomes_conjuntos]
for label, val in zip(labels, flat_00):
    print(f"  {label} = {val:.4f}")

# Pesos W: shape (9, num_rules)
# num_rules = 4 (pequeno para o exemplo ficar legível)
num_rules = 4
np.random.seed(7)
W_rules = np.random.randn(C * NS, num_rules).astype(np.float32) * 0.5
# shape: (9, 4)

print(f"\nMatriz W de pesos (shape {W_rules.shape}):")
print(f"(9 entradas × {num_rules} regras)\n")
header = "          " + "".join([f"  Regra_{r}" for r in range(num_rules)])
print(header)
print("-" * 50)
for i, label in enumerate(labels):
    row = "  ".join([f"{W_rules[i,r]:+.4f}" for r in range(num_rules)])
    print(f"{label:12}: {row}")

# Cálculo da ativação das regras
# flat: (1, 4, 4, 9)  @  W_rules: (9, 4)  →  (1, 4, 4, 4)
rules_pre = flat @ W_rules    # (1, 4, 4, 4)
rules_act = np.maximum(0, rules_pre)  # ReLU

print(f"\nAtivação das regras para pixel (0,0):")
print(f"  y_r = ReLU( flat_00 @ W_rules )\n")
for r in range(num_rules):
    soma = sum(flat_00[k] * W_rules[k, r] for k in range(C * NS))
    ativacao = max(0, soma)
    print(f"  Regra {r}: Σ(μ × W) = {soma:+.4f}  →  ReLU = {ativacao:.4f}")

print(f"\nShape após inferência: {rules_act.shape}")
print("(batch=1, H=4, W=4, num_rules=4)\n")


# ==============================================================================
# PASSO 3: DEFUZZIFICAÇÃO (método do Centroide / Média Ponderada)
# ==============================================================================
# Método: Centro de Gravidade (COG) — Centroide
#
# Na lógica fuzzy clássica:
#
#         Σ_r ( ativação_r × valor_consequente_r )
# saída = ─────────────────────────────────────────
#                   Σ_r ( ativação_r )
#
# Na nossa implementação (TSK):
#   - Os "valores consequentes" são os pesos aprendíveis W_defuzz
#   - A divisão pelo somatório é substituída por tanh (normalização suave)
#   - Isso é equivalente ao centroide quando as ativações são não-negativas
#
# Implementação:
#   saída_canal_c = Σ_r  ativação_r × W_defuzz[r, c]  +  bias_c
#   saída_canal_c = tanh(saída_canal_c)
# ==============================================================================

print("=" * 60)
print("PASSO 3: DEFUZZIFICAÇÃO (Centroide / Média Ponderada)")
print("=" * 60)

print("""
Centroide clássico:
         Σ_r ( y_r × c_r )
  saída = ──────────────────
               Σ_r ( y_r )

Na implementação (TSK generalizado):
  saída_c = tanh( Σ_r  y_r × W_defuzz[r, c]  +  bias_c )

Onde W_defuzz são os "valores consequentes" aprendíveis.
""")

# Pesos de defuzzificação: (num_rules, C_original)
np.random.seed(13)
W_defuzz = np.random.randn(num_rules, C).astype(np.float32) * 0.3
bias_defuzz = np.zeros(C, dtype=np.float32)

print(f"Matriz W_defuzz (shape {W_defuzz.shape}):")
print(f"({num_rules} regras → {C} canais de saída)\n")
header2 = "          " + "".join([f"  Canal_{c}" for c in range(C)])
print(header2)
print("-" * 40)
for r in range(num_rules):
    row = "  ".join([f"{W_defuzz[r,c]:+.4f}" for c in range(C)])
    print(f"Regra {r}:   {row}")

# Cálculo da defuzzificação para pixel (0,0)
rules_00 = rules_act[0, 0, 0, :]   # (4,)  ativações das 4 regras

print(f"\nCalculando defuzzificação do pixel (0,0):")
print(f"Ativações das regras: {rules_00}\n")

for c in range(C):
    numerador = sum(rules_00[r] * W_defuzz[r, c] for r in range(num_rules))
    soma_ativ  = sum(rules_00) + 1e-8

    # Centroide clássico (para referência)
    centroide = numerador / soma_ativ

    # Implementação com tanh
    linear = numerador + bias_defuzz[c]
    saida  = np.tanh(linear)

    print(f"Canal {c}:")
    print(f"  Σ(y_r × W_defuzz[r,{c}]) = {numerador:+.4f}")
    print(f"  Σ(y_r)                   = {soma_ativ:.4f}")
    print(f"  Centroide clássico        = {centroide:+.4f}")
    print(f"  tanh({linear:.4f})         = {saida:+.4f}  ← valor final\n")

# Aplica para todos os pixels
defuzz_all = np.tanh(rules_act @ W_defuzz + bias_defuzz)  # (1, 4, 4, 3)

print(f"Shape após defuzzificação: {defuzz_all.shape}")
print("(batch=1, H=4, W=4, C=3)  ← shape original restaurado!\n")


# ==============================================================================
# PASSO 4: CONEXÃO RESIDUAL
# ==============================================================================

print("=" * 60)
print("PASSO 4: CONEXÃO RESIDUAL")
print("=" * 60)

print("""
Fórmula:
  saída_final = imagem_original + saída_fuzzy

Por que isso é importante?
  - No início do treino, W_rules e W_defuzz são aleatórios
  - A saída fuzzy é basicamente ruído
  - A conexão residual garante que o gradiente flua mesmo assim
  - Gradiente de saída_final em relação à imagem_original = 1 (sempre)
""")

saida_final = imagem[0] + defuzz_all[0]  # (4, 4, 3)

print("Pixel (0,0) — comparação:")
print(f"{'Canal':<8} {'Original':>10} {'Fuzzy':>10} {'Final':>10}")
print("-" * 42)
for c in range(C):
    orig  = imagem[0, 0, 0, c]
    fuzzy = defuzz_all[0, 0, 0, c]
    final = saida_final[0, 0, c]
    print(f"Canal {c}:   {orig:>10.4f} {fuzzy:>10.4f} {final:>10.4f}")


# ==============================================================================
# RESUMO FINAL
# ==============================================================================

print("\n" + "=" * 60)
print("RESUMO — TRANSFORMAÇÕES DO PIXEL (0,0)")
print("=" * 60)
print(f"""
Entrada (3 valores):
  Canal 0 = {imagem[0,0,0,0]:.4f}
  Canal 1 = {imagem[0,0,0,1]:.4f}
  Canal 2 = {imagem[0,0,0,2]:.4f}

Após Fuzzificação (3 canais × 3 conjuntos = 9 valores):
  {flat_00}

Após Inferência TSK (4 regras):
  {rules_act[0,0,0,:]}

Após Defuzzificação (3 valores):
  {defuzz_all[0,0,0,:]}

Após Residual (3 valores finais):
  {saida_final[0,0,:]}
""")

print("Shapes em cada etapa:")
print(f"  Entrada:          (1, 4, 4, 3)")
print(f"  Fuzzificação:     (1, 4, 4, 3, 3)   ← +1 dimensão (conjuntos fuzzy)")
print(f"  Achatar:          (1, 4, 4, 9)       ← C × num_sets")
print(f"  Inferência TSK:   (1, 4, 4, 4)       ← num_rules")
print(f"  Defuzzificação:   (1, 4, 4, 3)       ← volta ao C original")
print(f"  Residual:         (1, 4, 4, 3)       ← shape idêntico à entrada")