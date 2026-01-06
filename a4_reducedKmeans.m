%% reducedKmeans para reducción de dimensionalidad 

% Este script carga el archivo train.mat, que contiene los datos de 
% entrenamiento en la estructura 'train' (con campos X_train y y_train).
% Para cada una de las 10 etiquetas, se aplica kmeans para obtener 50 
% centroides representativos. El resultado (500 muestras en total) se guarda 
% en el archivo train_reduced.mat.

% Cargar el archivo de entrenamiento
load('train_reducted.mat');  
% Extraer las variables
X = train.X_train;  % Dimensiones: [N_train, 100]
y = train.y_train;  % Dimensiones: [N_train, 1]

% Parámetro: número de clusters por etiqueta
clustersPerLabel = 100;

% Obtener las etiquetas únicas
uniqueLabels = unique(y);

% Inicializar variables para acumular las muestras reducidas
X_reduced = [];
y_reduced = [];

% Procesar cada etiqueta
for i = 1:length(uniqueLabels)
    currentLabel = uniqueLabels(i);
    fprintf('Procesando etiqueta %d...\n', currentLabel);
    
    % Seleccionar las muestras de la etiqueta actual
    idx = (y == currentLabel);
    X_label = X(idx, :);
    
    % Verificar si hay suficientes muestras para kmeans
    if size(X_label, 1) < clustersPerLabel
        warning('La etiqueta %d tiene menos muestras que clusters deseados. Se usarán todas las muestras.', currentLabel);
        centroids = X_label;
    else
        % Aplicar kmeans para obtener 'clustersPerLabel' centroides
        % Se utilizan 5 replicaciones para mayor estabilidad
        [~, centroids] = kmeans(X_label, clustersPerLabel, 'Replicates', 5, 'Display', 'final');
    end
    
    % Acumular los centroides y las etiquetas correspondientes
    X_reduced = [X_reduced; centroids];
    y_reduced = [y_reduced; repmat(currentLabel, size(centroids, 1), 1)];
end

% Mostrar las dimensiones resultantes (deben ser 200 x 100 y 200 x 1)
fprintf('Dimensiones de X_reduced: [%d, %d]\n', size(X_reduced,1), size(X_reduced,2));
fprintf('Dimensiones de y_reduced: [%d, 1]\n', size(y_reduced,1));

% Guardar la nueva estructura de datos reducida
train_reducted.X_train = X_reduced;
train_reducted.y_train = y_reduced;
train=train_reducted;
save('train200.mat', 'train');

fprintf('Reducción completada. Datos guardados en train_reduced.mat\n');
