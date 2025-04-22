import random
import math
import matplotlib
import matplotlib.axes
import matplotlib.pyplot
import numpy
import scipy
import scipy.optimize

def generate_individ(len):
    """Генерация одной особи 
    (из одной хромосомы) длиной len"""
    individ = []

    for i in range(0, len):
        bit=random.choice([0, 1])
        individ += [bit]

    return individ;

def generate_population(num, len):
    """Сгенерировать популяцию размером num и длиной хромосом len"""
    population=[]
    for i in range(0, num): population += [generate_individ(len)]
    return population

def bin2dec(chrom):
    """Из бит в десятичное"""
    dec=int()
    for i in range(0, len(chrom)):
        dec += math.pow(chrom[i] * 2, len(chrom) - i - 1)
    return dec

def phenotype(chrom, start, end, num):
    """Фенотип (значение) из хромосомы (бит)"""
    return start + bin2dec(chrom) * ((end-start) / num)

def x_to_y(f, x) -> list:
    """Массив x в массив y"""
    y = []
    for v in x: y.append(f(v))
    return y

def reproduction(population: list, f, start, end, num): 
    
    """Оператор репродукции"""
    #todo
    intermediate_population = [];

    individ_values = []

    # сумма всех значений и значение для каждой особи
    for v in population:
        x = phenotype(v, start, end, num)
        y = f(x)
        individ_values += [y]

    # смещаем, чтобы не было отрицательных значений, считаем сумму
    min_poten = min(individ_values)
    offset = 0 - min_poten

    for i in range(len(individ_values)):
        individ_values[i] += offset

    values_sum = sum(individ_values)

    # все особи одинаковы
    if values_sum == 0:
        return population.copy()
    
    potentials = []    

    # подсчет потенциала для каждой особи
    for i in range(len(individ_values)):
        val = individ_values[i]
        prob = val / values_sum
        potentials += [prob]

    sum_potentials = sum(potentials)

    # выбор такого же кол-ва особей
    for i in range(len(population)):

        # крутите барабан
        shot = random.random() * sum_potentials
        individ_num = 0
        tmp_sum = float()
        
        # определяем куда попали барабаном
        for j in range(len(population)):
            individ_num = j
            tmp_sum += potentials[individ_num]
            if (tmp_sum >= shot):
                break
            
        # добавляем выбранную особь в промежуточную популяцию
        individ = population[individ_num]
        intermediate_population += [individ]

    return intermediate_population

def crossingover(pop : list, p, f, start, end, num):

    """Кроссинговер - скрещивание популяции"""
    pop_copy = pop.copy()
    pop_result = []

    pop_len = len(pop)

    while (len(pop_copy) > 0): 
        p1 = pop_copy.pop(random.randrange(len(pop_copy)))
        p2 = pop_copy.pop(random.randrange(len(pop_copy)))

        pop_result += [p1.copy(), p2.copy()]

        if (random.random() < p):
            #выполняем скрещивание
            bit = random.randrange(len(p1))

            part = p1[bit:]
            p1[bit:] = p2[bit:]
            p2[bit:] = part

            pop_result += [p1, p2]

    def cmp(v): return f(phenotype(v, start, end, num))

    pop_result.sort(key = cmp, reverse=True)

    pop_result = pop_result[:pop_len]

    return pop_result
        
def mutation(pop: list, p):
    """Выполнить мутацию"""
    pop_copy = pop.copy()

    for v in pop_copy:
        if (random.random() < p):
            bit = random.randrange(len(v))
            v[bit] = 1 if v[bit] == 0 else 0

    return pop_copy

def draw_plot(f, start, end, num = int(math.pow(2, 15))):
    """Нарисовать график функции"""
    x = numpy.linspace(start, end, num)
    y = []    
    for i in x: y += [f(i)];
    matplotlib.pyplot.scatter(x, y, s=0.5)

def draw_population(f, pop, start, end, num):
    """Нарисовать точки популяции"""
    for individ in pop:
        x=phenotype(individ, start, end, num)
        matplotlib.pyplot.scatter(x, f(x), s=15, color="red")

def population_phenotypes(phenotype, start, end, num, pop):
    """Получить фенотипы (значения) для популяции"""
    xs = []
    for v in pop:
        xs += [phenotype(v, start, end, num)]
    return xs

def genetic(
        f, 
        start, 
        end, 
        chrom_len, 
        mutation_prob, 
        crossing_prob, 
        population_size, 
        localization, 
        max_steps,
        is_draw):
    
    """Генетический алгоритм"""

    combinations_num = 2**chrom_len

    population = generate_population(population_size, chrom_len)

    if (is_draw):
        matplotlib.pyplot.figure(num = "Начальное поколение")
        draw_plot(f, start, end)
        draw_population(f, population, start, end, combinations_num)

    # matplotlib.pyplot.legend()

    step = 1

    while True:

        if step + 1 > max_steps:
            break

        step += 1

        reproducted_population = reproduction(population, f, start, end, combinations_num)
        cross_population = crossingover(reproducted_population, crossing_prob, f, start, end, combinations_num)
        mutated_population = mutation(cross_population, mutation_prob)

        population = mutated_population

        xs = population_phenotypes(phenotype, start, end, combinations_num, mutated_population)

        ys = x_to_y(f, xs)

        y_max = max(ys)        
        x_best = xs[ys.index(y_max)]

        if (is_draw):
            matplotlib.pyplot.figure(num = f"Поколение {step}")
            draw_plot(f, start, end)
            draw_population(f, population, start, end, combinations_num)
            matplotlib.pyplot.scatter([x_best], [y_max], s=25, color="green", label=f"Максимум = {y_max}")
            matplotlib.pyplot.legend()

        if (abs(min(xs) - max(xs)) < localization):
            break 

    x = max(xs)

    print("Работа генетического алгоритма завершена")
    print(f"Размер популяции {population_size}")
    print(f"Вероятность кроссинговера {crossing_prob}")
    print(f"Вероятность мутации {mutation_prob}")
    print(f"Ограничивающая локализация решений: {localization}")
    print(f"Выполнено {step} шагов")

    print(f"Итоговая локализация решений - [{min(xs)}; {max(xs)}] длиной {abs(min(xs) - max(xs))}")
    print(f"Решение: f({x}) = {f(x)}")

    # проверка решения заведомо надежным решением
    def neg_x(x): 
        return - f(x[0]) 

    real_solve_x, = scipy.optimize.brute(neg_x, [[start, end]])
    real_solve_y = f(real_solve_x)

    print(f"Реальное решение: f({real_solve_x}) = {real_solve_y}")

    deviation = abs(x - real_solve_x)

    print(f"Отклонение x: {deviation}")
    print(f"Отклонение y: {abs(f(x) - f(real_solve_x))}")
    print()

    if (is_draw):        
        matplotlib.pyplot.show(block = True)

    return x, f(x), xs, step


# Вариант 4
# Функция Sin(2x)/x2
def f(x: float): return math.sin(2*x) / math.pow(x, 2)

# Диапазон решений - x[-20,-3.1] - длина 16.9, 16900 значений для точности 0,001
# следовательно, длина хромосомы достаточна - 15 бит
start = -20.
end = -3.1
bits = 15
#combinations_num = int(math.pow(2, bits))

#вероятность мутации
mutation_prob = 0.001

# вероятность потомства
crossing_prob = 0.5

# размер популяции
population_size = 100

# отрезок локализации решения
localization = 0.01

# максимальное кол-во шагов
max_steps = 100

# запуск алгоритма
genetic(
    f, 
    start = start, 
    end = end, 
    chrom_len = bits, 
    mutation_prob = mutation_prob, 
    crossing_prob = crossing_prob, 
    population_size = 100, 
    localization = localization,  
    max_steps = max_steps, 
    is_draw = False)

# exit()

print("популяция 1000 особей")
genetic(
    f, 
    start = start, 
    end = end, 
    chrom_len = bits,
    mutation_prob = mutation_prob, 
    crossing_prob = crossing_prob, 
    population_size = 1000, 
    localization = localization,  
    max_steps = max_steps, 
    is_draw = False)

print("пониженная вероятность кроссинговера = 0.2")
genetic(
    f, 
    start = start, 
    end = end, 
    chrom_len = bits,
    mutation_prob = mutation_prob, 
    crossing_prob = 0.2, 
    population_size = population_size, 
    localization = localization,  
    max_steps = max_steps, 
    is_draw = False)

print("повышенная вероятность мутации = 0.1")
genetic(
    f, 
    start = start, 
    end = end, 
    chrom_len = bits,
    mutation_prob = 0.1, 
    crossing_prob = crossing_prob, 
    population_size = population_size, 
    localization = localization,  
    max_steps = max_steps, 
    is_draw = False)

print("популяция 10 особей")
genetic(
    f, 
    start = start, 
    end = end, 
    chrom_len = bits,
    mutation_prob = mutation_prob, 
    crossing_prob = crossing_prob, 
    population_size = 10, 
    localization = localization,  
    max_steps = max_steps, 
    is_draw = False)