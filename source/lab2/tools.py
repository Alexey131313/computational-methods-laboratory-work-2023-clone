from random import uniform

def rand_vector(n: int, _min: float, _max: float):
    #Возвращает список размера n со случайными значениями от _min до _max.
    result = []
    for i in range(n):
        result.append(uniform(_min, _max))
    return result


def rand_matrix(_str: int, _col: int, _min: float, _max: float):
    #Возвращает двумерный список размера (_str)x(_col) со случайными значениями от _min до _max.
    result = []
    for i in range(_str):
        _str = []
        for j in range(_col):
            _str.append(uniform(_min, _max))
        result.append(_str)
    return result


def combin_matrix(left_matrix, right_matrix):
    #Объединяет две матрицы с одинаковым количеством строк в одну и возвращает ее.
    result = []
    str_size = len(left_matrix)
    col_size_left = len(left_matrix[0])
    col_size_right = len(right_matrix[0])
    for i in range(str_size):
        _str = []

        for j in range(col_size_left):
            _str.append(left_matrix[i][j])

        for j in range(col_size_right):
            _str.append(right_matrix[i][j])

        result.append(_str)
    return result


def create_p_matrix(a_vector, b_matrix):
    #Возвращает матрицу P для решения задачи оптимизации (Элементы в ней не могут быть больше 1).
    result = []
    n = len(a_vector)

    for i in range(n):
        _str = [a_vector[i]]
        for j in range(1, n):
            res = _str[j - 1] * b_matrix[i][j - 1]
            if res > 1:
                res = 1
            _str.append(res)
        result.append(_str)
    return result