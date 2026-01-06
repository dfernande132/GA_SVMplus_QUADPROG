%% Script para visualizar 5 imágenes de cada etiqueta del train_reduced.mat

% Cargar el archivo train_reduced.mat (se asume que contiene la estructura train_reduced)
l%oad('train200.mat');  % Contiene: train_reduced.X_train (500 x 784) y train_reduced.y_train (500 x 1)

% Extraer los datos
X = train.X_train;
y = train.y_train;

% Número de imágenes a mostrar por cada etiqueta
nImages = 5;
pixelImagen = floor(sqrt(size(X, 2)));
% Obtener las etiquetas únicas
uniqueLabels = unique(y);

% Para cada etiqueta, selecciona 5 imágenes y muéstralas en una figura
for i = 1:length(uniqueLabels)
    currentLabel = uniqueLabels(i);
    % Encontrar índices de las muestras que tienen la etiqueta actual
    indices = find(y == currentLabel);
    
    % En caso de que por alguna razón haya menos de 5 muestras, usamos el mínimo disponible
    selectedIndices = indices(randperm(length(indices), nImages));
    
    % Crear una nueva figura para la etiqueta actual
    figure;
    for j = 1:nImages
        subplot(1, nImages, j);
        % Convertir la fila del vector a una imagen cuadrada
        img = reshape(X(selectedIndices(j), :), pixelImagen, pixelImagen);
        imshow(img, []);  % '[]' ajusta la visualización al rango de valores de la imagen
        title(sprintf('Label %d', currentLabel));
    end
    % Título general para la figura (requiere Matlab R2018b o posterior; si no, se puede omitir)
    sgtitle(sprintf('Etiqueta %d - %d imágenes', currentLabel, nImages));
end
