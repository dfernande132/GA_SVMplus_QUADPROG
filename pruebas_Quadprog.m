 clc;clear; close all;

%% *** Generar Datos de Ejemplo ***
numfv = 400;  % Numero de muestras
dimFv = 12;   % Caracteristicas en fv
dimFvStar = 2; % Caracteri­sticas en fvStar

% Datos aleatorios normalizados
fv = normalize(randn(numfv, dimFv));      
fvStar = normalize(randn(numfv, dimFvStar)); 
lbl = sign(randn(numfv, 1));  % Etiquetas (+1/-1)
Y = diag(lbl);
% Parametros del modelo SVM+
Cparam = 0.1;  % Vector de C
gammaParam = 0.1;
sgmPlus = 1;
sgmStar = 0.5;

%% *** Calcular Matrices de Kernel ***
KMatrPlus = exp(-pdist2(fv, fv).^2 / (2 * sgmPlus^2));
KMatrStar = exp(-pdist2(fvStar, fvStar).^2 / (2 * sgmStar^2));
%yLab = lbl * lbl';

%% *** Resolver con solve ***
tic;
alphaPlus = optimvar('alphaPlus', numfv, 'LowerBound', 0);
betaPlus = optimvar('betaPlus', numfv, 'LowerBound', 0);

probPlus = optimproblem;
probPlus.Objective = -sum(alphaPlus) + ...
    0.5 * sum(sum((alphaPlus * alphaPlus') .* (Y * KMatrPlus * Y))) + ...
    (0.5 / gammaParam) * sum(sum(((alphaPlus + betaPlus - Cparam) * (alphaPlus + betaPlus - Cparam)').* KMatrStar));

probPlus.Constraints.cons1 = sum(alphaPlus .* lbl) == 0;
probPlus.Constraints.cons2 = sum(alphaPlus + betaPlus - Cparam) == 0;

optPlus = optimoptions(probPlus, 'MaxIterations', 800, 'Display', 'off');
[solSolve, valPlus, ~, ~, ~] = solve(probPlus, 'Options', optPlus);
alphaSolve = solSolve.alphaPlus(:);
betaSolve = solSolve.betaPlus(:);
t1=toc;
fprintf('Tiempo ejecucion solve: %.1f seg\n',t1);

%% *** Resolver con quadprog ***
tic;
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

% Restricción de igualdad: sum(alpha .* y) = 0
Aeq = [lbl', zeros(1, numfv)];
beq = 0;

% Restricción de igualdad: sum(alpha + beta - C) = 0
Aeq = [Aeq; ones(1, numfv), ones(1, numfv)];
beq = [beq; Cparam * numfv];

% Restricción de desigualdad: 
Aineq = [];
bineq = [];

% Restricciones de no negatividad
lb = zeros(2*numfv,1);
ub = [];

% Resolver con quadprog
options = optimoptions('quadprog','Display','off');
[z_sol, fval, exitflag, output] = quadprog(H, f, Aineq, bineq,  Aeq, beq, lb, ub, [], options);

% Extraer soluciones para alpha y beta
alphaQuadprog = z_sol(1:numfv);
betaQuadprog  = z_sol(numfv+1:end);

% Verificar si quadprog encuentra solucion
if exitflag ~= 1
    fprintf('Warning: quadprog no encontro una solucion optima. Codigo de salida: %d\n', exitflag);
    alphaQuadprog = zeros(numfv, 1);
end
t2=toc;
fprintf('Tiempo ejecucion quadprog: %.1f seg\n',t2);

%% *** Comparar Resultados ***
fprintf('Comparando resultados de alpha+ obtenidos con solve y quadprog...\n');

if length(alphaSolve) == length(alphaQuadprog)
    diffAlpha = norm(alphaSolve - alphaQuadprog);
    diffBeta = norm(betaSolve - betaQuadprog);
    diffAlpha = mean(abs(alphaSolve - alphaQuadprog));
    diffBeta = mean(abs(betaSolve - betaQuadprog));
    diffVal = abs(fval-valPlus);
    fprintf('Diferencia en alphas: %.6f\n', diffAlpha);
    if diffAlpha < 1e-3
        fprintf('Los resultados son similares (diferencia < 0.001).\n');
    else
        fprintf('hay diferencias en los lagrangianos alpha.\n');
    end
    fprintf('Diferencia en betas: %.6f\n', diffBeta);
    if diffBeta < 1e-3
        fprintf('Los resultados son similares (diferencia < 0.001).\n');
    else
        fprintf('hay diferencias en los lagrangianos beta.\n');
    end

else
    fprintf('Diferencia en resultado de alphaSolve (%d) y alphaQuadprog (%d)\n', ...
            length(alphaSolve), length(alphaQuadprog));
end
