klines = 0;

while (klines <= 0)
    klines = input("Введите кол-во килострок:\n");
end

% чтение данных проектов
data = read_data('COCOMO_data.txt');
% количество проектов, на которых проходит обучение
learning_projects_num = 40;

learning_data_indexes = randperm(length(data), learning_projects_num);
learning_data = cell(1, learning_projects_num);
check_data = cell(1, length(data) - learning_projects_num);

learn_cnt = 1;
check_cnt = 1;

for i = 1:length(data)
    if (any(ismember(learning_data_indexes, i)))
        learning_data{learn_cnt} = data{i}; 
        learn_cnt = learn_cnt + 1;
    else
        check_data{check_cnt} = data{i}; 
        check_cnt = check_cnt + 1;
    end
end

% % данные для обучения - 40 шт
% learning_data = {data{1:20}, data{41:60}};
% 
% % данные для проверки - 20 шт
% check_data = {data{21:40}};

f = @(x)fitness(x(1), x(2), learning_data);
cross = @(a, b) flat_crossover(a, b);

res = float_genetic(f, 100, 2, 20, 0.5, 0.1, cross, 1, 100, 0, 1);

a = res{end}{1}(1);
b = res{end}{1}(2);

costs = efm(a, b, klines);
fprintf("Затраты на проект размером %f килострок: %f человекомесяцев\n", klines, costs);

errors = zeros(1, length(res));

for i = 1:length(res)
    errors(i) = res{i}{2};
end

figure("Name", "Уменьшение ошибки на каждом этапе");
plot(errors, "-o");
legend("Ошибка ED - Евклидово расстояние");

learn_data_errors = zeros(length(learning_data), 2);
nums = strings(length(learning_data), 1);

for i = 1:length(learning_data)
    s = learning_data{i};
    learn_data_errors(i, 1) = ed(s.EF_C, s.EF);
    learn_data_errors(i, 2) = ed(efm(a,b,s.L), s.EF);
    nums(i) = string(s.num);
end

figure("Name", "Ошибки для обучающих данных");
bar(nums, learn_data_errors);
legend("Ошибка ED COCOMO", "Ошибка ED ГА");
xlabel("Номера проектов");
ylabel("Ошибка ED");
yscale log

check_data_errors = zeros(length(check_data), 2);
nums = strings(length(check_data), 1);

for i = 1:length(check_data)
    s = check_data{i};
    check_data_errors(i, 1) = ed(s.EF_C, s.EF);
    check_data_errors(i, 2) = ed(efm(a,b,s.L), s.EF);
    nums(i) = string(s.num);
end

figure("Name", "Ошибки для тестовых данных");
bar(nums, check_data_errors);
legend("Ошибка ED COCOMO", "Ошибка ED ГА");
xlabel("Номера проектов");
ylabel("Ошибка ED");
yscale log

check_error = fitness(a, b, check_data);

% for i = 1:length(check_data)
%     s = check_data
%     error = f(a, b, )
% end

% s = fitness(1, 1, learning_data);

 % float_genetic( ...
 %    f, ...
 %    population_size, ...
 %    nodes_count, ...
 %    max_steps, ...
 %    cross_prob, ...
 %    mutation_prob, ...
 %    cross, ...
 %    coord_from, ...
 %    coord_to)



% Вариант 4: ГА Веществ. вектор ED арифметич арифметич рулетка

% Фитнесс - функция: эвклидово расстояние
function v = fitness(a, b, learn_data)
    len = length(learn_data);
    sum = 0;

    for i = 1:len
        s = learn_data{i};
        ef = s.EF;
        l = s.L;
        efmi = efm(a, b, l);
        subsum = (ef - efmi) .^ 2;
        sum = sum + subsum;
    end

    v = sqrt(sum ./ len);    
end

function v = ed(ef, efmi)
    len = length(ef);
    sum = 0;

    for i = 1:len        
        subsum = (ef(i) - efmi(i)) .^ 2;
        sum = sum + subsum;
    end

    v = sqrt(sum ./ len);  
end

% Расчет сложности проекта по кол-ву строк, 
% коэффициентам a и b и кол-ву килострок l
function e = efm(a, b, l)
    e = (l .^ b) .* a;
end


% Чтение файла с данными для обучения
function data = read_data(fileName)

    data = {};
    
    fileID = fopen(fileName, 'r');
    formatSpec = '%f';   
    a = fscanf(fileID, formatSpec);

    row_len = 7;

    for i = 1:((length(a)/row_len))
        offset = row_len * (i - 1);

        s.num = a(offset + 1);
        s.L = a(offset + 2);
        s.EF = a(offset + 3);
        s.EF_C = a(offset + 4);

        data{length(data) + 1} = s;
    end
end

function res = float_genetic( ...
    f, ...
    population_size, ...
    nodes_count, ...
    max_steps, ...
    cross_prob, ...
    mutation_prob, ...
    cross, ...
    a_coord_from, ...
    a_coord_to, ...
    b_coord_from, ...
    b_coord_to)

    population = generate_population( ...
        population_size, ...
        nodes_count, ...
        a_coord_from, ...
        a_coord_to, ...
        b_coord_from, ...
        b_coord_to);

    res = cell(1, max_steps);

    for i = 1:max_steps

        if (length(population) < population_size)
            a=1;
        end

        parents = reproduction(population, f);  

        % Сохраняем лучшую особь
        best = best_individ(f, population);
        best_ind = population{best};

        if (best > 1)
            start = population(1:best-1);
        else
            start = cell(0,1);
        end

        if (best < length(population))
            ending = population(best+1:end);
        else
            ending = cell(0,1);
        end

        without_best_pop = [start;ending];

        % рождение и мутация детей
        children = crossover(parents, cross_prob, cross);
        children = mutation(children, mutation_prob);
    
        % редукция
        new_population = [without_best_pop; children];              

        population = reduction(f, new_population, population_size - 1);
        population = [{best_ind}; population];

        best = best_individ(f, population);
        best_ind = population{best};        

        fprintf("Шаг %d: лучшие значения:%s:%f\n", i, mat2str(best_ind), f(best_ind));        
        % fprintf("Длина маршрута: %d\n", f(best_ind));

        res{i} = {best_ind, f(best_ind)};
    end

    % best = best_individ(f, population);
    % res = {population{best}; f(population{best})};
end

function num = best_individ(f, pop)

    min_val = intmax();
    num = 0;

    for i = 1:length(pop)
        val = f(pop{i});
        if (val < min_val)
            min_val = val;
            num = i;
        end
    end
end

% Плоский кроссовер
function child = flat_crossover(p1, p2)
    dim = length(p1);
    child = zeros(dim);

    for i = 1:dim

        diff = abs(p1(i) - p2(i));

        gap = diff .* 0.25;

        child(i) = rand_range(min(p1(i), p2(i)) - gap, max(p2(i), p1(i)) + gap);
    end
end

% Мутация
function mutated_pop = mutation(pop, prob)

    for i = 1:length(pop)
        if (rand() < prob)
            val = pop{i};
            for j = 1:length(val)
                val(j) = rand_range(val(j) - 1, val(j) + 1);            
            end
            pop{i} = val;
        end
    end

    mutated_pop = pop;
end

% function draw_route(path, coords)
%     x = zeros(1, length(path));
%     y = zeros(1, length(path));
% 
%     for i = 1:length(path)
%         node = path(i);
%         x(i) = coords(node, 1);
%         y(i) = coords(node, 2);
%     end
% 
%     figure;
%     plot(x, y, '-x');
% 
% end

% Генерация начальной популяции
function pop = generate_population(pop_size, ind_size, from, to, from2, to2)
    pop = cell(pop_size, 1);
    % pop = zeros(pop_size, ind_size);
    for i = 1:pop_size
        ind = zeros(1, ind_size);

        ind(1) = rand_range(from, to);
        ind(1) = rand_range(from2, to2);

        % for j = 1:ind_size
        %     ind(j) = 
        % end
        pop{i} = ind;
    end
end


% случайное число в диапазоне
function ret = rand_range(from, to)
    ret = (to-from) .* rand() + from;
end

% Редукция
function reducted_pop = reduction(f, pop, len)
    valind = configureDictionary("double", "cell");
    cnt = 0;

    arr = {{}};
    
    for i = 1:length(pop)
        val = f(pop{i});       
        if cnt > 0
            if (valind.isKey(val))
                arr = valind.lookup(val);
            else
                arr = {{}};
            end             
        else
            arr = {{}};
        end

        subarr = arr{1};
        subarr{length(subarr) + 1} = pop{i}; 
        arr{1} = subarr;

        % if (cnt == 0)
        %     val2ind = dictionary(val, arr);
        % end
        valind = valind.insert(val, arr);       
        cnt = cnt + 1;
    end

    reducted_pop = {};    

    while (length(reducted_pop) < len & ~isempty(valind.keys()))
        minval = min(valind.keys);
        arr = valind.lookup(minval);

        unique = configureDictionary("double", "double");

        for j = 1:length(arr{1})

            ind = arr{1}{j};

            % is_member = ismember([ind], cell2mat(reducted_pop));

            if (~unique.isKey(ind))
                reducted_pop = [reducted_pop; ind];
                unique = unique.insert(ind, 1);
            end
        end
        valind = valind.remove(minval);
    end
end

% Репродукция
function intermediate_population = reproduction(population, f) 
    
    pop_size = length(population);

    % Оператор репродукции
    intermediate_population = cell(pop_size, 1);

    individ_values = zeros(pop_size, 1);

    % сумма всех значений и значение для каждой особи
    for i = 1:pop_size
        val = f(population{i});
        individ_values(i) = val;
    end

    values_sum = sum(individ_values);

    % все особи одинаковы
    if values_sum == 0
        intermediate_population = population;
        return
    end

    potentials = zeros(pop_size, 1); 

    % подсчет потенциала для каждой особи
    % потенциалы отрицательны, т.к. ищем минимум
    for i = 1:length(individ_values)
        val = individ_values(i);
        prob = -(val / values_sum);
        potentials(i) = prob;
    end

    min_potential = min(potentials);

    % сдвигаем потенциалы в положительную часть
    for i = 1:length(individ_values)
        potentials(i) = potentials(i) - min_potential;
    end

    sum_potentials = sum(potentials);

     % выбор такого же кол-ва особей
    for i = 1:length(population)

         % крутите барабан
        shot = rand_range(0, sum_potentials);
        individ_num = 0;
        tmp_sum = 0;

        % определяем куда попали барабаном
        for j = 1:length(population)
            individ_num = j;
            tmp_sum = tmp_sum + potentials(individ_num);
            if (tmp_sum >= shot)
                break
            end
        end

        % добавляем выбранную особь в промежуточную популяцию
        individ = population{individ_num};
        intermediate_population{i} = individ;                  
    end
end

% Скрещивание
function pop = crossover(parents, prob, cros)
    len = length(parents);
    pop = cell(0, 1);

    for i = 1:len
        p1 = parents{round(rand_range(1, len))};
        p2 = parents{round(rand_range(1, len))};

        if (rand() < prob)
            child = cros(p1, p2);    
            pop{length(pop) + 1, 1} = child;
        end
    end
end
