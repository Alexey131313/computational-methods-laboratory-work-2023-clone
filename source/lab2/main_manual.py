# main_manual.py

# Импорт необходимых библиотек и модулей
import numpy as np  # Импорт библиотеки NumPy для работы с массивами и матрицами
import alg as alg  # Импорт созданного модуля alg.py
import matplotlib.pyplot as plt  # Импорт библиотеки для построения графиков
from tkinter import Tk, Label, Button, Entry, W, Toplevel, END, Frame, BOTTOM, LEFT, RIGHT, messagebox, ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import mplcursors
import time
import progress_indicator as load
import sys
    
def Table(n, e, a_min, a_max, bd_min, bd_max, b_min, b_max, k):
    
    s = 0
    s1 = 0
    s2 = 0
    s3 = 0
    s4 = 0
    s5 = 0
    s6 = 0
    s7 = 0
    s8 = 0

    # Проведение экспериментов для указанного числа раз (e)
    for i in range(1, e+1):
        
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

        # Выполнение задачи назначения для различных алгоритмов
        task_matrix = P
        ass_by_Hun_min = h.TaskAssignment(task_matrix, 'Hungary_min', v)
        ass_by_Hun_max = h.TaskAssignment(task_matrix, 'Hungary_max', v)
        ass_by_greedy = h.TaskAssignment(task_matrix, 'greedy', v)
        ass_by_t = h.TaskAssignment(task_matrix, 'Thrifty', v)
        ass_by_gt = h.TaskAssignment(task_matrix, 'Greedy_Thrifty', v)
        ass_by_tg = h.TaskAssignment(task_matrix, 'Thrifty_Greedy', v)
        ass_by_tkg = h.TaskAssignment(task_matrix, 'TkG', v, k)
        ass_by_ctg = h.TaskAssignment(task_matrix, 'CTG', v, k)
        ass_by_gk = h.TaskAssignment(task_matrix, 'Gk', v, k)

        # Суммирование результатов
        s += ass_by_Hun_max.max_cost
        s1 += ass_by_greedy.max_cost
        s2 += ass_by_Hun_min.min_cost
        s3 += ass_by_t.max_cost
        s4 += ass_by_gt.max_cost
        s5 += ass_by_tg.max_cost
        s6 += ass_by_tkg.max_cost
        s7 += ass_by_ctg.max_cost
        s8 += ass_by_gk.max_cost

    # Расчет средних значений результатов
    s = s / e
    s1 = s1 / e
    s2 = s2 / e
    s3 = s3 / e
    s4 = s4 / e
    s5 = s5 / e
    s6 = s6 / e
    s7 = s7 / e
    s8 = s8 / e

    # Создание данных для вывода в таблицу
    table_data = [
        ["Венгерский max", s, ""],
        ["Жадный", s1, abs(s-s1)/s],
        ["Венгерский min", s2, abs(s-s2)/s],
        ["Бережливый", s3, abs(s-s3)/s],
        ["Жадный-Бережливый", s4, abs(s-s4)/s],
        ["Бережливый-Жадный", s5, abs(s-s5)/s],
        ["TkG", s6, abs(s-s6)/s],
        ["CTG", s7, abs(s-s7)/s],
        ["Gk", s8, abs(s-s8)/s],
    ]
    
    # Вывод таблицы
    h.print_table(table_data)

    # Запись таблицы в файл и вывод на экран
    with open('data.txt', 'w') as file:
        file.write(str(table_data))
    
# Функция для генерации и построения экспериментальных результатов
def Experiment(n, a_min, a_max, bd_min, bd_max, b_min, b_max, k):
    pi = load.ProgressIndicator(root)
    
    x = [i + 1 for i in range(n)]
    pi.update_progress(2)
    mu = n//2
    
    pi.set_label("Генерация матрицы...")
    p = alg.rand_p_matrix(size=n, min_start_sugar=a_min, max_start_sugar=a_max, min_maturation=b_min, max_maturation=b_max, min_degradation=bd_min, max_degradation=bd_max)
    pi.update_progress(4)
    # Венгерский минимальный
    pi.set_label("Вычисление значений Венгерского алгоритма...")
    r1, indices1, y1 = alg.Hungary_min(p)
    pi.update_progress(10)

    # Венгерский максимальный
    r2, indices2, y2 = alg.Hungary_max(p)
    pi.update_progress(16)
    
    # Жадный
    pi.set_label("Вычисление значений Жадного алгоритма...")
    r3, indices3, y3 = alg.Greedy(p)
    pi.update_progress(22)

    # Бережливый
    pi.set_label("Вычисление значений Бережливого алгоритма...")
    r4, indices4, y4 = alg.Thrifty(p)
    pi.update_progress(28)
    
    # Жадно-бережливый алгоритм
    pi.set_label("Вычисление значений Жадно-бережливого алгоритма...")
    r5, indices5, y5 = alg.Greedy_thrifty(p, mu)
    pi.update_progress(36)

    # Бережливо-жадный алгоритм
    pi.set_label("Вычисление значений Бережливо-жадного алгоритма...")
    r6, indices6, y6 = alg.Thrifty_greedy(p, mu)
    pi.update_progress(44)

    #TkG
    pi.set_label("Вычисление значений TkG алгоритма...")
    r7, indices7, y7 = alg.TkG(p, mu, 2)
    print("Размерность массива x:", len(x))
    print("Размерность массива summa:", len(y7))
    pi.update_progress(52)
    k = 1
    for i in range(3, mu):
        r, ind, res = alg.TkG(p, mu, i)
        if (r7 < r):
            r7 = r
            indices7 = ind
            y7 = res
            k = i
    pi.update_progress(60)

    #CTG
    pi.set_label("Вычисление значений CTG алгоритма...")
    r8, indices8, y8 = alg.CTG(p, mu, 2)
    pi.update_progress(68)
    k2 = 1
    for i in range(3, n):
        r, ind, res = alg.CTG(p, mu, 2)
        if (r8 < r):
            r8 = r
            indices8 = ind
            y8 = res
            k2 = i
    pi.update_progress(76)

    #Gk
    pi.set_label("Вычисление значений Gk алгоритма...")
    r9, indices9, y9 = alg.Gk(p, mu, 2)
    pi.update_progress(84)
    k3 = 1
    for i in range(3, n):
        r, ind, res = alg.Gk(p, mu, i)
        if (r9 < r):
            r9 = r
            indices9 = ind
            y9 = res
            k3 = i
    pi.update_progress(92)
    
    S = []

    pi.set_label("Сохраняем результаты...")
    # print('\n')
    S.append("Целевая функция Венгерского минимума: " + str(format(r1, '.4f')))
    S.append("Целевая функция Венгерского максимума: " + str(format(r2, '.4f')))
    S.append("Целевая функция Жадного алгоритма: " + str(format(r3, '.4f')))
    S.append("Целевая функция Бережливого алгоритма: " + str(format(r4, '.4f')))
    S.append("Целевая функция Жадно-бережливого алгоритма: " + str(format(r5, '.4f')))
    S.append("Целевая функция Бережливо-жадного алгоритма: " + str(format(r6, '.4f')))
    S.append("Целевая функция T(" + str(k) + ")G алгоритма: " + str(format(r7, '.4f')))
    S.append("Целевая функция CTG(k=" + str(k2) + ") алгоритма: " + str(format(r8, '.4f')))
    S.append("Целевая функция Gk(k=" + str(k3) + ") алгоритма: " + str(format(r9, '.4f')))
    pi.set_label("Готово!")
    pi.update_progress(100)
    pi.close()
    
    # Построение графика
    show(n, x, y1, y2, y3, y4, y5, y6, y7, y8, y9)
    
        
def on_enter(event):
    ax = event.inaxes
    if ax is not None:
        ax.figure.canvas.mpl_connect('motion_notify_event', on_motion)

def on_leave(event):
    ax = event.inaxes
    if ax is not None:
        ax.figure.canvas.mpl_disconnect('motion_notify_event')

def on_press(event):
    ax = event.inaxes
    if event.button == 1:  # Проверяем, что нажата левая кнопка мыши
        if ax is not None:
            ax._pan_start = event.x, event.y
            ax._start_x, ax._start_y = event.x, event.y  # Сохраняем начальные координаты для первого события движения мыши
            ax.figure.canvas.mpl_connect('motion_notify_event', on_motion)  # Подключаем обработчик движения мыши
            ax.figure.canvas.mpl_connect('axes_leave_event', on_leave)  # Подключаем обработчик выхода из области графика

def on_release(event):
    ax = event.inaxes
    if ax is not None:
        ax._pan_start = None
        ax.figure.canvas.mpl_disconnect('motion_notify_event')  # Отключаем обработчик движения мыши
        ax.figure.canvas.mpl_disconnect('axes_leave_event')  # Отключаем обработчик выхода из области графика

def on_motion(event):
    ax = event.inaxes
    if ax is not None and hasattr(ax, '_pan_start') and ax._pan_start is not None:
        start_x, start_y = ax._pan_start  # Получаем начальные координаты мыши
        dx = event.x - ax._start_x
        dy = event.y - ax._start_y
        xlim_min, xlim_max = ax.get_xlim()
        ylim_min, ylim_max = ax.get_ylim()
        
        # Получаем текущий размер окна графика
        width, height = ax.figure.get_size_inches()
        
        # Вычисляем коэффициент коррекции скорости перемещения на основе размера окна
        correction_factor = 0.0001 * max(width, height)
        
        # Применяем коэффициент коррекции к смещению
        dx *= correction_factor
        dy *= correction_factor
        
        # Вычисляем новые границы осей графика
        xlim_new_min = xlim_min - dx
        xlim_new_max = xlim_max - dx
        ylim_new_min = ylim_min - dy
        ylim_new_max = ylim_max - dy
        
        # Устанавливаем новые границы осей графика
        ax.set_xlim(xlim_new_min, xlim_new_max)
        ax.set_ylim(ylim_new_min, ylim_new_max)
        ax.figure.canvas.draw_idle()
        
        # Обновляем начальные координаты для следующего события движения мыши
        ax._start_x, ax._start_y = event.x, event.y



# Функция для построения графика
def show(n, x, y, y1, y2, y3, y4, y5, y6, y7, y8):
    
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.plot(x, y, label='Венгерский мин', color='blue', linestyle='-')
    ax1.plot(x, y1, label='Венгерский макс', color='green', linestyle='--')
    ax1.plot(x, y2, label='Жадный', color='red', linestyle=':', marker='o', markersize=2)
    ax1.plot(x, y3, label='Бережливый', color='purple', linestyle='-.')
    ax1.plot(x, y4, label='Жадно-бережливый', color='orange', linestyle='-', marker='o', markersize=2)
    ax1.plot(x, y5, label='Бережливо-жадный', color='brown', linestyle='--')
    ax1.plot(x, y6, label='TkG', color='black', linestyle='-.', marker='^', markersize=4)
    ax1.plot(x, y7, label='CTG', color='gray', linestyle='--', marker='s', markersize=4)
    ax1.plot(x, y8, label='Gk', color='magenta', linestyle=':', marker='D', markersize=3)

    # Добавление вертикальной черты
    ax1.axvline(x=len(x) / 2, color='gray', linestyle='--', linewidth=1, label='x = ν')
    
    ax1.set_title('Сравнение алгоритмов')
    ax1.set_xlabel('T, этап обработки')
    ax1.set_ylabel('S, суммарное содержание сахара')
    ax1.legend()
    ax1.grid(linewidth=0.2)
    
    m2 = [0 for i in range(n)]
    m1 = [abs(y1[i] - y[i]) / y[i] for i in range(n)]
    m3 = [abs(y1[i] - y2[i]) / y2[i] for i in range(n)]
    m4 = [abs(y1[i] - y3[i]) / y3[i] for i in range(n)]
    m5 = [abs(y1[i] - y4[i]) / y4[i] for i in range(n)]
    m6 = [abs(y1[i] - y5[i]) / y5[i] for i in range(n)]
    m7 = [abs(y1[i] - y6[i]) / y6[i] for i in range(n)]
    m8 = [abs(y1[i] - y7[i]) / y7[i] for i in range(n)]
    m9 = [abs(y1[i] - y8[i]) / y8[i] for i in range(n)]

    ax2.set_title("Сравнение ошибок")
    ax2.set_xlabel("Этап обработки")
    ax2.set_ylabel("Ошибка")
    ax2.plot(x, m1,  label='Венгерский мин', color='blue', linestyle='-')
    ax2.plot(x, m2,  label='Венгерский макс', color='green', linestyle='--')
    ax2.plot(x, m3, label='Жадный', color='red', linestyle=':', marker='o', markersize=2)
    ax2.plot(x, m4, label='Бережливый', color='purple', linestyle='-.')
    ax2.plot(x, m5, label='Жадно-бережливый', color='orange', linestyle='-', marker='o', markersize=2)
    ax2.plot(x, m6,  label='Бережливо-жадный', color='brown', linestyle='--')
    ax2.plot(x, m7, label='TkG', color='black', linestyle='-.', marker='^', markersize=4)
    ax2.plot(x, m8, label='CTG', color='gray', linestyle='--', marker='s', markersize=4)
    ax2.plot(x, m9, label='Gk', color='magenta', linestyle=':', marker='D', markersize=3)
    ax2.legend()
    
    # Добавление аннотаций к точкам графика при наведении курсора
    mplcursors.cursor(hover=True).connect("add", lambda sel: sel.annotation.set_text(f"Функция: {sel.artist.get_label()}"))
    
    # Привязываем функции к событиям мыши на холсте графика
    fig.canvas.mpl_connect('button_press_event', on_press)
    fig.canvas.mpl_connect('button_release_event', on_release)
    fig.canvas.mpl_connect('motion_notify_event', on_motion)
    fig.canvas.mpl_connect('scroll_event', on_scroll)
    
    # Создаем отдельное окно для графика
    graph_window = Toplevel(root)
    graph_window.title("Графики")
    
    # Устанавливаем размер окна с графиком
    graph_window.state("zoomed")

    # Отображение графика в новом окне
    canvas = FigureCanvasTkAgg(fig, master=graph_window)
    canvas.draw()
    canvas.get_tk_widget().pack(side='top', fill='both', expand=1)
    

# Функция для вызова экспериментов по кнопке
def run_experiments():
    n = int(entry_n.get())
    e = int(entry_e.get())
    a_min = float(entry_a_min.get())
    a_max = float(entry_a_max.get())
    bd_min = float(entry_bd_min.get())
    bd_max = float(entry_bd_max.get())
    b_min = float(entry_b_min.get())
    b_max = float(entry_b_max.get())

    try:
        k = int(entry_k.get())
        if k < 1 or k > n // 2:
            raise ValueError
    except ValueError:
        k = 2
        entry_k.delete(0, END)
        entry_k.insert(0, '2')

    # Вызов функций для построения графика и создания таблицы
    Experiment(n, a_min, a_max, bd_min, bd_max, b_min, b_max, k)
    #Table(n, e, a_min, a_max, bd_min, bd_max, b_min, b_max, k)
    #theor_solution()
    
def reset_values():
    for key, value in default_values.items():
        # Accessing entry fields directly
        entry = globals()[f'entry_{key}']
        entry.delete(0, END)
        entry.insert(0, value)
        
def on_scroll(event):
    axtemp = event.inaxes
    x_min, x_max = axtemp.get_xlim()
    y_min, y_max = axtemp.get_ylim()
    x_range = (x_max - x_min) / 10
    y_range = (y_max - y_min) / 10

    if event.button == 'up':
        axtemp.set(xlim=(x_min + x_range, x_max - x_range), ylim=(y_min + y_range, y_max - y_range))
    elif event.button == 'down':
        axtemp.set(xlim=(x_min - x_range, x_max + x_range), ylim=(y_min - y_range, y_max + y_range))
    plt.draw()

def on_closing():
    if messagebox.askokcancel("Закрыть", "Вы уверены, что хотите закрыть приложение?"):
        root.destroy()
        sys.exit()
        
if __name__ == '__main__':
    # Открываем окно
    root = Tk()
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.title("Решение задачи дискретной оптимизации")
    root.geometry("600x200")
    root.resizable(False, False)

    # Фрейм для элементов слева
    left_frame = Frame(root)
    left_frame.grid(row=0, column=0, padx=10, pady=10, sticky='w')

    # Фрейм для элементов справа
    right_frame = Frame(root)
    right_frame.grid(row=0, column=1, padx=10, pady=10, sticky='e')

    # Добавляем поля для ввода данных слева
    Label(left_frame, text="Число этапов:").grid(row=0, column=0, sticky='w')
    entry_n = Entry(left_frame, width=10)
    entry_n.grid(row=0, column=1, sticky='w')

    Label(left_frame, text="Число экспериментов:").grid(row=1, column=0, sticky='w')
    entry_e = Entry(left_frame, width=10)
    entry_e.grid(row=1, column=1, sticky='w')

    Label(left_frame, text="Минимальная сахаристость a:").grid(row=2, column=0, sticky='w')
    entry_a_min = Entry(left_frame, width=10)
    entry_a_min.grid(row=2, column=1, sticky='w')

    Label(left_frame, text="Максимальная сахаристость a:").grid(row=3, column=0, sticky='w')
    entry_a_max = Entry(left_frame, width=10)
    entry_a_max.grid(row=3, column=1, sticky='w')

    # Добавляем поля для ввода данных справа
    Label(right_frame, text="b дозревания минимальное (b > 1):").grid(row=0, column=0, sticky='e')
    entry_bd_min = Entry(right_frame, width=10)
    entry_bd_min.grid(row=0, column=1, sticky='e')

    Label(right_frame, text="b дозревания максимальное (b > 1):").grid(row=1, column=0, sticky='e')
    entry_bd_max = Entry(right_frame, width=10)
    entry_bd_max.grid(row=1, column=1, sticky='e')

    Label(right_frame, text="b увядания минимальное (b < 1):").grid(row=2, column=0, sticky='e')
    entry_b_min = Entry(right_frame, width=10)
    entry_b_min.grid(row=2, column=1, sticky='e')

    Label(right_frame, text="b увядания максимальное (b < 1):").grid(row=3, column=0, sticky='e')
    entry_b_max = Entry(right_frame, width=10)
    entry_b_max.grid(row=3, column=1, sticky='e')

    Label(right_frame, text="k").grid(row=4, column=0, sticky='e')
    entry_k = Entry(right_frame, width=10)
    entry_k.grid(row=4, column=1, sticky='e')

    # Устанавливаем стандартные значения для полей
    default_values = {
        'n': '50',
        'e': '15',
        'a_min': '0.05',
        'a_max': '0.15',
        'bd_min': '1.01',
        'bd_max': '1.07',
        'b_min': '0.93',
        'b_max': '0.99',
        'k': '2'
    }

    for key, value in default_values.items():
        vars()[f'entry_{key}'].insert(0, value)

    # Кнопки внизу по центру окна
    buttons_frame = Frame(root)
    buttons_frame.grid(row=1, column=0, pady=10, sticky='s')

    run_button = Button(buttons_frame, text="Запустить эксперименты", command=run_experiments)
    run_button.grid(row=0, column=0, padx=5)

    reset_button = Button(buttons_frame, text="Сбросить значения", command=reset_values)
    reset_button.grid(row=0, column=1, padx=5)

    # Чтобы разместить кнопки внизу по центру, можем использовать columnspan=2 и sticky='n'
    buttons_frame.grid_rowconfigure(0, weight=1)
    buttons_frame.grid_columnconfigure(0, weight=1)
    buttons_frame.grid_columnconfigure(1, weight=1)


    # Запускаем окно
    root.mainloop()




