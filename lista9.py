import csv
import math
import statistics

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

caminho = r"C:\Users\kaleg\Downloads\creditcard.csv\creditcard.csv"

# ========================================
# 0) LER CSV
# ========================================
dados = []
with open(caminho, newline="", encoding="utf-8") as f:
    leitor = csv.reader(f)
    cabecalho = next(leitor)  # pega o header
    for linha in leitor:
        dados.append(linha)

print("\nCabeçalho:", cabecalho)
print("Primeiras 3 linhas:")
for linha in dados[:3]:
    print(linha)

# ========================================
# Converter tudo para float (menos a classe)
# ========================================
dados_convertidos = []
for linha in dados:
    nova = []
    for i, valor in enumerate(linha):
        if i == len(linha)-1:
            nova.append(int(valor))  # classe é inteira
        else:
            nova.append(float(valor))  
    dados_convertidos.append(nova)

dados = dados_convertidos

# ========================================
# 2) TRATAR VALORES AUSENTES
# (substituir por mediana manualmente)
# ========================================

def coluna(dados, i):
    return [linha[i] for linha in dados]

for i in range(len(cabecalho)-1):  # menos a classe
    col = coluna(dados, i)
    # valores não nulos
    sem_nan = [c for c in col if not math.isnan(c)]
    mediana = statistics.median(sem_nan)

    # substituir ausentes
    for linha in dados:
        if math.isnan(linha[i]):
            linha[i] = mediana

print("\nValores ausentes tratados.")

# ========================================
# 3) REMOVER REDUNDÂNCIA
# (colunas com variância zero)
# ========================================

cols_para_remover = []
for i in range(len(cabecalho)-1):
    col = coluna(dados, i)
    if statistics.pvariance(col) == 0:
        cols_para_remover.append(i)

print("\nColunas redundantes removidas (variância zero):", cols_para_remover)

# remover as colunas
for linha in dados:
    for i in sorted(cols_para_remover, reverse=True):
        del linha[i]

# atualizar cabecalho
for i in sorted(cols_para_remover, reverse=True):
    del cabecalho[i]

# ========================================
# 4) TRATAMENTO DE OUTLIERS (IQR)
# ========================================

def tratar_outliers(col):
    Q1 = statistics.quantiles(col, n=4)[0]
    Q3 = statistics.quantiles(col, n=4)[2]
    IQR = Q3 - Q1
    limite_inf = Q1 - 1.5 * IQR
    limite_sup = Q3 + 1.5 * IQR
    return limite_inf, limite_sup

for i in range(len(cabecalho)-1):
    col = coluna(dados, i)
    lim_inf, lim_sup = tratar_outliers(col)

    for linha in dados:
        if linha[i] < lim_inf:
            linha[i] = lim_inf
        elif linha[i] > lim_sup:
            linha[i] = lim_sup

print("\nOutliers tratados.")

# ========================================
# 5) NORMALIZAÇÃO COM STANDARD SCALER
# ========================================

X = [linha[:-1] for linha in dados]   # todas menos classe
y = [linha[-1] for linha in dados]    # classe

scaler = StandardScaler()
X = scaler.fit_transform(X)

print("\nNormalização concluída.")

# ========================================
# 6) Correlação e multicolinearidade
# (feito manualmente por matriz de correlação)
# ========================================

def correlacao(v1, v2):
    return statistics.correlation(v1, v2)

print("\nMatriz de correlação (parcial):")
for i in range(3):
    print([round(correlacao(coluna(X, i), coluna(X, j)), 3) for j in range(3)])

# ========================================
# 7) Codificação de variáveis
# (não existe categórica no creditcard.csv)
# ========================================

print("\nNenhuma variável categórica encontrada. (OK)")

# ========================================
# 8) Balanceamento COM SMOTE
# ========================================

sm = SMOTE()
X_bal, y_bal = sm.fit_resample(X, y)

print("\nBalanceamento SMOTE:")
print("Antes:", {0: y.count(0), 1: y.count(1)})
print("Depois:", {0: list(y_bal).count(0), 1: list(y_bal).count(1)})

# ========================================
# 9) DIVISÃO TREINO–TESTE (estratificada)
# ========================================

X_train, X_test, y_train, y_test = train_test_split(
    X_bal, y_bal, test_size=0.2, stratify=y_bal, random_state=42
)

print("\nTreino:", len(X_train))
print("Teste :", len(X_test))
