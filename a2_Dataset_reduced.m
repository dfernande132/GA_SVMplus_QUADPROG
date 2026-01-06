%% Script para reducir imágenes de train_original.mat y test_original.mat
% Selecciona únicamente las imágenes con etiquetas 5 y 8, reduce cada imagen
% de 28x28 a 10x10 y vectoriza a 100 elementos. El resultado se guarda en:
%   - train_reducted.mat (estructura train_reducted con X_train y y_train)
%   - test_reducted.mat  (estructura test_reducted con X_test y y_test)

%% 1. Cargar los datos originales de entrenamiento y test
load('train_original.mat');  % Carga la estructura "train" con X_train (60000x784) y y_train (60000x1)
load('test_original.mat');   % Carga la estructura "test"  con X_test y y_test

%% 2. Procesar el conjunto de entrenamiento (seleccionar etiquetas 5 y 8)
selectedIdx_train = find(train.y_train == 5 | train.y_train == 8);
numSelected_train = length(selectedIdx_train);

% Prealocar la nueva matriz para X_train reducida: [n_train x 100]
new_X_train = zeros(numSelected_train, 100);

% Procesar cada imagen seleccionada
for i = 1:numSelected_train
    % Extraer la imagen original (vector de 784) y reorganizarla a 28x28
    img = reshape(train.X_train(selectedIdx_train(i), :), [28, 28]);
    % Reducir la imagen a 10x10
    img_resized = imresize(img, [10, 10]);
    % Vectorizar la imagen reducida (resultado: 100 elementos) y asignarla a la fila i
    new_X_train(i, :) = img_resized(:)';
end

% Extraer las etiquetas correspondientes
new_y_train = train.y_train(selectedIdx_train);

% Crear la nueva estructura para entrenamiento
train_reducted.X_train = new_X_train;
train_reducted.y_train = new_y_train;
train=train_reducted;
% Guardar la estructura en un archivo .mat
save('train_reducted.mat', 'train');
fprintf('TRAIN: Se han seleccionado %d imágenes con etiquetas 5 y 8 y reducido su tamaño a 10x10 (vectorizadas a 100 elementos).\n', numSelected_train);
fprintf('Datos guardados en train_reducted.mat\n');

%% 3. Procesar el conjunto de test (seleccionar etiquetas 5 y 8)
selectedIdx_test = find(test.y_test == 5 | test.y_test == 8);
numSelected_test = length(selectedIdx_test);

% Prealocar la nueva matriz para X_test reducida: [n_test x 100]
new_X_test = zeros(numSelected_test, 100);

% Procesar cada imagen seleccionada
for i = 1:numSelected_test
    % Extraer la imagen original (vector de 784) y reorganizarla a 28x28
    img = reshape(test.X_test(selectedIdx_test(i), :), [28, 28]);
    % Reducir la imagen a 10x10
    img_resized = imresize(img, [10, 10]);
    % Vectorizar la imagen reducida (resultado: 100 elementos)
    new_X_test(i, :) = img_resized(:)';
end

% Extraer las etiquetas correspondientes
new_y_test = test.y_test(selectedIdx_test);

% Crear la nueva estructura para test
test_reducted.X_test = new_X_test;
test_reducted.y_test = new_y_test;
test=test_reducted;

% Guardar la estructura en un archivo .mat
save('test_reducted.mat', 'test');
fprintf('TEST: Se han seleccionado %d imágenes con etiquetas 5 y 8 y reducido su tamaño a 10x10 (vectorizadas a 100 elementos).\n', numSelected_test);
fprintf('Datos guardados en test_reducted.mat\n');
