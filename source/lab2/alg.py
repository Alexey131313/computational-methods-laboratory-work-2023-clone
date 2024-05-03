#alg.py
import tools
import numpy as np
from scipy.optimize import linear_sum_assignment
from random import uniform
import time
import threading
import multiprocessing

# Функция для создания и вывода таблицы результатов
def print_table(data):
    # Вывод заголовка таблицы
    header = ["Алгоритм", "Среднее значение S", "Относительная погрешность"]
    print(f"{header[0]:<20} {header[1]:<20} {header[2]:<20}")

    # Вывод данных таблицы
    for row in data:
        print(f"{row[0]:<20} {row[1]:<20} {row[2]:<20}")
        
def createRndMatrix(n, a_min, a_max, bd_min, bd_max, b_min, b_max):
    v = n//2
    # Создание матрицы P на основе сгенерированных значений
    P = np.zeros((n, n))
    for m in range(n):
        for k in range(n):
            if k == 0:
                P[m][k] = np.random.uniform(a_min, a_max)
                    
            else:
                if k < v:
                    b = np.random.uniform(bd_min, bd_max)
                else:
                    b = np.random.uniform(b_min, b_max)
                P[m][k] = P[m][k-1] * b
    return P

def rand_p_matrix(size: int, min_start_sugar: float, max_start_sugar: float,
                 min_maturation: float, max_maturation: float, min_degradation: float, max_degradation: float):

    sugar_cols = size // 2
    degradation_cols = size - sugar_cols
    a_vector = tools.rand_vector(size, min_start_sugar, max_start_sugar) # стартовая сахаристость
    sugar_matrix = tools.rand_matrix(size, sugar_cols, min_maturation, max_maturation) # часть матрицы для дозаривания
    degradation_matrix = tools.rand_matrix(size, degradation_cols, min_degradation, max_degradation)# это деградирующая часть
    b_matrix = tools.combin_matrix(sugar_matrix, degradation_matrix)
    p_matrix = tools.create_p_matrix(a_vector, b_matrix)
    return p_matrix

def Hungary_min(p_matrix):
    #Возвращает результат и список-перестановку целевой функции, поиск худшего результата с помощью венгерского алгоритма.
    row_indices, col_indices = linear_sum_assignment(np.array(p_matrix).transpose()) # Список "row_indices" содержит индексы строк,
                                                               # а список "col_indices" - индексы столбцов, которые образуют оптимальное назначение.
    result = 0
    summa = [] # S на каждом шаге
    
    for i in range(len(row_indices)):
        result += p_matrix[col_indices[i]][row_indices[i]]   # итоговая сумма
        summa.append(result)
    # for i in range(len(row_indices)):
    #     row_indices[col_indices[i]] = i
    return result, col_indices, summa


def Hungary_max(p_matrix):
    #Возвращает результат и список-перестановку целевой функции, поиск лучшего результата с помощью венгерского алгоритма.
    reverse_p_matrix = np.copy(p_matrix) #обратная матрица
    for i in range(len(p_matrix)):
        max_elem = np.max(p_matrix[i])
        for j in range(len(p_matrix)):
            reverse_p_matrix[i][j] = -1 * p_matrix[i][j] + max_elem # сводим максимизацию к минимизации
                                                                    #(в каждой строке берем максимальный элемент и вычитаем из него все остальные элементы строки)

    row_indices, col_indices = linear_sum_assignment(np.array(reverse_p_matrix).transpose())
    result = 0
    summa = []

    for i in range(len(row_indices)):
        result += p_matrix[col_indices[i]][row_indices[i]]
        summa.append(result)

    # for i in range(len(row_indices)):
    #     col_indices[row_indices[i]] = i
    return result, col_indices, summa

def Greedy(p_matrix: list):
    #Возвращает результат и список-перестановку целевой функции, поиск результата с помощью жадного алгоритма.
    result = 0
    indices = []
    took = []
    summa = []

    for j in range(len(p_matrix)): # идем по столбцам
        col_max = 0
        col_max_index: int
        for i in range(len(p_matrix)): # по строкам данного столбца
            is_took = False

            for k in range(len(took)):
                if took[k] == i:
                    is_took = True
                    break

            if is_took: # если такой элемент уже брали, то переходим к следующему
                continue

            if p_matrix[i][j] > col_max:
                col_max = p_matrix[i][j]
                col_max_index = i
        result += col_max
        summa.append(result)
        indices.append(col_max_index)
        took.append(col_max_index)
    return result, indices, summa


def Thrifty(p_matrix: list):
    #Возвращает результат и список-перестановку целевой функции, поиск результата с помощью бережливого алгоритма.
    result = 0
    indices = []
    took = []
    summa = []
    
    for j in range(len(p_matrix)):
        col_min = 1000
        col_min_index: int
        for i in range(len(p_matrix)):
            is_took = False

            for k in range(len(took)):
                if took[k] == i:
                    is_took = True
                    break

            if is_took:
                continue

            if p_matrix[i][j] < col_min:
                col_min = p_matrix[i][j]
                col_min_index = i
        result += col_min
        summa.append(result)
        indices.append(col_min_index)
        took.append(col_min_index)
    return result, indices, summa

def Thrifty_greedy(p_matrix: list, saving_steps: int):
    #Возвращает результат и список-перестановку целевой функции, поиск результата с помощью бережливо-жадного алгоритма.
    #saving_steps - количество шагов в режиме сбережения, далее будет жадный режим.
    result = 0
    indices = []
    took = []
    summa = []
    saving_steps_completed = 0

    for j in range(len(p_matrix)):

        col_min = 10
        col_min_index: int

        col_max = 0
        col_max_index: int
        saving = False

        if saving_steps_completed < saving_steps:
            saving = True

        for i in range(len(p_matrix)):
            is_took = False

            for k in range(len(took)):
                if took[k] == i:
                    is_took = True
                    break

            if is_took:
                continue

            if saving and p_matrix[i][j] < col_min:
                col_min = p_matrix[i][j]
                col_min_index = i

            if not saving and p_matrix[i][j] > col_max:
                col_max = p_matrix[i][j]
                col_max_index = i

        if saving:
            result += col_min
            summa.append(result)
            indices.append(col_min_index)
            took.append(col_min_index)

        else:
            result += col_max
            summa.append(result)
            indices.append(col_max_index)
            took.append(col_max_index)

        saving_steps_completed += 1

    return result, indices, summa



def Greedy_thrifty(p_matrix: list, greedy_steps: int):
    #Возвращает результат и список-перестановку целевой функции, поиск результата с помощью жадно-бережливого алгоритма.\n
    #greedy_steps - количество шагов в режиме жадности, далее будет бережливый режим.
    result = 0
    indices = []
    took = []
    summa = []
    greedy_steps_completed = 0

    for j in range(len(p_matrix)):

        col_min = 1000
        col_min_index: int

        col_max = 0
        col_max_index: int
        greedy = False
        
        if greedy_steps_completed < greedy_steps:
            greedy = True

        for i in range(len(p_matrix)):
            is_took = False

            for k in range(len(took)):
                if took[k] == i:
                    is_took = True
                    break

            if is_took:
                continue

            if not greedy and p_matrix[i][j] < col_min:
                col_min = p_matrix[i][j]
                col_min_index = i

            if greedy and p_matrix[i][j] > col_max:
                col_max = p_matrix[i][j]
                col_max_index = i

        if greedy:
            result += col_max
            summa.append(result)
            indices.append(col_max_index)
            took.append(col_max_index)

        else:
            result += col_min
            summa.append(result)
            indices.append(col_min_index)
            took.append(col_min_index)

        greedy_steps_completed += 1

    return result, indices, summa

def TkG(p_matrix: list, saving_steps: int,  k: int):
    result = 0
    indices = []
    took = []
    summa = []
    saving_steps_completed = 0
    n = len(p_matrix)
    for j in range(n):

        col_min = 10
        col_min_index: int

        col_max = 0
        col_max_index: int
        saving = False

        tmp = [0 for i in range(n)]

        if saving_steps_completed < saving_steps:
            saving = True

        for i in range(len(p_matrix)):
            is_took = False

            for l in range(len(took)):
                if took[l] == i:
                    is_took = True
                    break

            if is_took:
                continue

            tmp[i] = p_matrix[i][j]

        ts = sorted(tmp)

        if saving:
            r = ts[min(k, n - j) - 1 + len(took)]
            result += r
            summa.append(result)
            indices.append(tmp.index(r))
            took.append(tmp.index(r))

        else:
            result += ts[n - 1]
            summa.append(result)
            indices.append(tmp.index(ts[n - 1]))
            took.append(tmp.index(ts[n - 1]))

        saving_steps_completed += 1

    return result, indices, summa

def CTG(p_matrix: list, saving_steps: int,  k: int):
    def gamma(k: int):
        if (1<=k<=mu-1):
            return n-2*mu+2*k+1
        elif (mu<=k<=2*mu-1):
            return n+2*mu-2*k
        else:
            return n-k+1
    n = len(p_matrix)
    mu = saving_steps
    indices = [0 for i in range(n)]
    summa = [0 for i in range(n)]
    for i in range(n):
        indices[gamma(i+1)-1] = i
        summa[i] = (summa[max(i-1, 0)] + p_matrix[gamma(i + 1) - 1][i])
    result = summa[n - 1]
    return result, indices, summa

def gamma(k, mu, n):
    if 1 <= k <= mu - 1:
        return n - 2 * mu + 2 * k + 1
    elif mu <= k <= 2 * mu - 1:
        return n + 2 * mu - 2 * k
    else:
        return n - k + 1

def Gk(p_matrix: list, saving_steps: int,  k: int):
    result = 0
    n = len(p_matrix)
    indices = []
    took = []
    summa = []
    tmp = []
    ts=[]
    for j in range(len(p_matrix)): # идем по столбцам
        if j % k == 0:
            tmp = [0 for i in range(n)]
            col_max = 0
            col_max_index: int
            for i in range(len(p_matrix)): # по строкам данного столбца
                is_took = False

                for l in range(len(took)):
                    if took[l] == i:
                        is_took = True
                        break

                if is_took: # если такой элемент уже брали, то переходим к следующему
                    continue

                tmp[i] = p_matrix[i][j]
            ts = sorted(tmp, reverse=True)
        result += p_matrix[tmp.index(ts[j % k])][j]
        summa.append(result)
        indices.append(tmp.index(ts[j % k]))
        took.append(tmp.index(ts[j % k]))
    return result, indices, summa
