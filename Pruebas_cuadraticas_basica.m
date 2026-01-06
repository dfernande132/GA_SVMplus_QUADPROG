clc;clear;

%% Datos del problema
numfv = 5000;  % Numero de muestras
dimFv = 12;   % Caracteristicas en fv
dimFvStar = 2; % Caracteri­sticas en fvStar

% Datos aleatorios 
fv = normalize(randn(numfv, dimFv));      
fvStar = normalize(randn(numfv, dimFvStar)); 
lbl = sign(randn(numfv, 1));  % Etiquetas (+1/-1)
Y = diag(lbl);

% Parametros del modelo SVM+
Cparam = 10;  % Vector de C
gammaParam = 0.1;
sgmPlus = 1;
sgmStar = 0.5;
% *** Calcular Matrices de Kernel ***
KMatrPlus = exp(-pdist2(fv, fv).^2 / (2 * sgmPlus^2));
KMatrStar = exp(-pdist2(fvStar, fvStar).^2 / (2 * sgmStar^2));
yLab = lbl * lbl';


% Multiplicadores de Lagrange aleatorios
alphaPlus = randi([0, 10], numfv, 1);  % Valores enteros entre 0 y 10
betaPlus  = randi([0, 10], numfv, 1);

%% 1. Cálculo directo de la función objetivo dual
% Se calcula:
tic;
f_direct = -sum(alphaPlus) + ...
    0.5 * sum(sum((alphaPlus * alphaPlus') .* (Y * KMatrPlus * Y))) + ...
    (0.5 / gammaParam) * sum(sum(((alphaPlus + betaPlus - Cparam) * (alphaPlus + betaPlus - Cparam)').* KMatrStar));
toc;

%% 2. Cálculo mediante la formulación estándar para quadprog
% Se define el vector z = [alpha; beta]
tic;
z = [alphaPlus; betaPlus];

% Construccion de la matriz H
H_alpha = Y * KMatrPlus * Y + (1/gammaParam) * KMatrStar;
H_beta  = (1/gammaParam) * KMatrStar;
H_cross = (1/gammaParam) * KMatrStar;

H = [H_alpha, H_cross;
     H_cross, H_beta];

% Construcción del vector f
oneVec = ones(numfv, 1);
K_sum = KMatrStar * oneVec;
f = [-oneVec - (1/gammaParam) * (Cparam * K_sum);
    - (1/gammaParam) * (Cparam * K_sum)];

% Construcción del término constante 
c = (0.5 / gammaParam) * (Cparam^2) * sum(KMatrStar(:));

% Cálculo de la función objetivo
f_quadprog = 0.5 * (z' * H * z) + f' * z + c;
toc;
%% Mostrar resultados
fprintf('Valor de la función objetivo (forma directa): %f\n', f_direct);
fprintf('Valor de la función objetivo (forma quadprog): %f\n', f_quadprog);
