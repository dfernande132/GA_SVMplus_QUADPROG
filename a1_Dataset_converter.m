%% Cargar los datos de entrenamiento
clc;clear;close all;
disp('Conversion del DataSet MNIST');
% Especifica las rutas a los archivos descomprimidos
trainImagesPath = 'MNIST_Dataset/train-images-idx3-ubyte/train-images-idx3-ubyte';
trainLabelsPath = 'MNIST_Dataset/train-labels-idx1-ubyte/train-labels-idx1-ubyte';
testImagesPath  = 'MNIST_Dataset/t10k-images-idx3-ubyte/t10k-images-idx3-ubyte';
testLabelsPath  = 'MNIST_Dataset/t10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte';

% Cargar los datos
[train_data, test_data] = loadMNISTData(trainImagesPath, trainLabelsPath, testImagesPath, testLabelsPath);

% Convertimos a matrices

% Para el conjunto de entrenamiento:
N_train = size(train_data.images, 3);
train.X_train = reshape(train_data.images, [], N_train)';  
train.y_train = double(train_data.labels(:));

% Para el conjunto de test:
N_test = size(test_data.images, 3);
test.X_test = reshape(test_data.images, [], N_test)';  
test.y_test = double(test_data.labels(:));

% Guardamos
save('train_original.mat', 'train');
save('test_original.mat', 'test');

disp('Conversion realizada. DataSet en test.mat y train.mat');