import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import easygui


def nd_simp(y, h):

    soma_ci_1 = 0.0
    soma_ci_2 = 0.0
    soma_ci_3 = 0.0

    for i in range(len(y)):
        if i == 0 or i == len(y)-1:
            soma_ci_1 += y.iloc[i]
        elif i % 3 == 0 and i != 0 and i != len(y)-1:
            soma_ci_2 += y.iloc[i]
        else:
            soma_ci_3 += y.iloc[i]

    return (3*h/8)*(soma_ci_1+(2*soma_ci_2)+(3*soma_ci_3))


def calc_integral(x, y):
    dados_result = {'Info': ['Impulso', 'Empuxo max (t)', 'Empuxo max (N)', 'Empuxo medio', 'Pontos', 'Duração', 'Nome'],
                    'Valor': [0.0, 0.0, 0.0, 0.0, '', '', '']}
    global df_result
    df_result = pd.DataFrame(dados_result)
    df_result.at[4, 'Valor'] = len(x)
    df_result.at[5, 'Valor'] = x.iloc[-1]-x.iloc[0]
    df_result.at[0, 'Valor'] = nd_simp(y, ((x.iloc[-1]-x.iloc[0])/(len(x)-1)))
    df_result.at[3, 'Valor'] = (
        1/(x.iloc[-1]-x.iloc[0]))*df_result.at[0, 'Valor']
    return None


def spline(xi, yi):
    n = len(xi)
    a = {k: v for k, v in enumerate(yi)}
    h = {k: xi[k+1]-xi[k] for k in range(n-1)}

    A = [[1]+[0]*(n-1)]
    for i in range(1, n-1):
        linha = np.zeros(n)
        linha[i-1] = h[i-1]
        linha[i] = 2 * (h[i-1] + h[i])
        linha[i+1] = h[i]
        A.append(linha)
    A.append([0]*(n-1)+[1])

    B = [0]
    for k in range(1, n-1):
        linha = 3 * (a[k+1]-a[k])/h[k] - 3 * (a[k] - a[k-1])/h[k-1]
        B.append(linha)
    B.append(0)

    c = dict(zip(range(n), np.linalg.solve(A, B)))

    b = {}
    d = {}
    for k in range(n-1):
        b[k] = (1/h[k])*(a[k+1]-a[k])-((h[k]/3)*(2*c[k]+c[k+1]))
        d[k] = (c[k+1]-c[k])/(3*h[k])

    s = {}
    for k in range(n-1):
        eq = f'{a[k]}{b[k]:+}*(x{-xi[k]:+}){c[k]                                            :+}*(x{-xi[k]:+})**2{d[k]:+}*(x{-xi[k]:+})**3'
        s[k] = {'eq': eq, 'dominio': [xi[k], xi[k+1]]}

    return s


def objective(x, a, b, c, d, e, f):
    return (a * x) + (b * x**2) + (c * x**3) + (d * x**4) + (e * x**5) + f


def open(arquivo):
    df = pd.read_excel(arquivo)
    yi = []
    xii = []
    data = df['Data'].tolist()
    h = df['Hora'].tolist()
    for i in range(len(df['Data'])):
        yi.append((float(df['Força'].iloc[i])))
        xii.append(float(df['Tempo de Queima'].iloc[i]))
    dados = {'Data': data, 'Hora': h, 'Força': yi, 'Tempo de Queima': xii}
    global df_dados
    df_dados = pd.DataFrame(dados)

    calc_integral(df['Tempo de Queima'], df['Força'])
    sn = spline(df['Tempo de Queima'], df['Força'])
    t = []
    pt = []
    for key, value in sn.items():
        def p(x):
            return eval(value['eq'])
        tx = np.linspace(*value['dominio'], 100)
        t.extend(tx)
        ptx = [p(x) for x in tx]
        pt.extend(ptx)

    df_result.at[2, 'Valor'] = max(pt)
    df_result.at[1, 'Valor'] = t[pt.index(max(pt))]
    label = arquivo.split('/')[-1]
    label2 = label.split('.')[0]
    ax.plot(t, pt, label=label2)
    df_result.at[6, 'Valor'] = label2
    print(df_result)
    print('\n')
    ax.plot(df['Tempo de Queima'], df['Força'], 'ko')

    # popt, _ = curve_fit(objective, df['Tempo de Queima'], df['Força'])
    # a, b, c, d, e, f = popt
    # x_line = np.arange(min(df['Tempo de Queima']), max(df['Tempo de Queima']), 0.1)
    # y_line = objective(x_line, a, b, c, d, e, f)
    # # ax.plot(x_line, y_line, '--', label=r'y = %.5f * x + %.5f * $x^2$ + %.5f * $x^3$ + %.5f * $x^4$ + %.5f * $x^5$ + %.5f' % (a, b, c, d, e, f))
    # ax.plot(x_line, y_line, '--', label=label2)
    # ax.set_ylim(min(df['Força']*-0.1), max(df['Força'])*1.2)


arquivos = [easygui.fileopenbox()]

fig, ax = plt.subplots(figsize=(10, 6))
for i in arquivos:
    open(i)

x1 = []
y1 = []
df1 = pd.read_excel(arquivos[0])
for i in range(len(df1['Data'])):
    y1.append((float(df1['Força'].iloc[i])))
    x1.append(float(df1['Tempo de Queima'].iloc[i]))

# x2 = []
# y2 = []
# df2 = pd.read_excel(arquivos[1])
# for i in range(len(df2['Data'])):
#     y2.append((float(df2['Força'].iloc[i])))
#     x2.append(float(df2['Tempo de Queima'].iloc[i]))

# xm = []
# ym = []
# for i in range(min(len(x1), len(x2))):
#     xm.append((x1[i] + x2[i]) / 2)
#     ym.append((y1[i] + y2[i]) / 2)

popt, _ = curve_fit(objective, x1, y1)
a, b, c, d, e, f = popt
print(f'({a} * t) + ({b} * t**2) + ({c} * t**3) + ({d} * t**4) + ({e} * t**5) + {f}')
x_line = np.arange(0, max(x1)+0.1, 0.1)
y_line = objective(x_line, a, b, c, d, e, f)

# calc_integral(pd.Series(x_line), pd.Series(y_line))

ax.plot(x_line, y_line, '--', label='Média')

ax.legend(fancybox=True, shadow=True, ncol=1)
ax.grid()
ax.set_xlabel('Tempo de Queima (s)')
ax.set_ylabel('Força (N)')
title = arquivos[0].split('/')[-1]
title2 = title.split('_')[0]
plt.title('Análise Conjunta - '+title2)
# plt.savefig('Analise de dados/Análise Conjunta - '+title2+'.png')
plt.show()
