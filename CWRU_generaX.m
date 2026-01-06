%% ------------------------------------------------------------
%  CWRU (12k Drive End, 1 HP)
%  - Construye X (features) e y (etiquetas) para Normal vs Fault
%  - Dibuja curvas temporales y perfiles por ventanas
% -------------------------------------------------------------

clear; clc;

%% Parámetros
rootDir     = 'CWRU_Dataset\';   % <--- AJUSTA esta ruta
Fs          = 12000;                  % 12 kHz (Drive End 12k)
Nwin        = 2048;                   % tamaño de ventana
hop         = Nwin/2;                 % 50% solape

%% Lista EXACTA de ficheros que indicaste
fileList = { ...
    'B007_1.mat','B014_1.mat','B021_1.mat','B028_1.mat', ...
    'IR007_1.mat','IR014_1.mat','IR021_1.mat','IR028_1.mat', ...
    'Normal_1.mat', ...
    'OR007@12_1.mat','OR007@3_1.mat','OR007@6_1.mat', ...
    'OR014@6_1.mat', ...
    'OR021@12_1.mat','OR021@3_1.mat','OR021@6_1.mat' ...
};

%% Construcción de X e y
X = []; y = []; Xstar = [];           % añadimos Xstar
file_ids = []; win_ids = [];
feat_names_base = feature_names_base();   % 15 nombres (A+C)
feat_names_star = feature_names_star();   % 15 nombres (B+D, globales)

rawSignals = struct();          % para gráficas (guarda algunas señales)
keepForPlot = {'Normal_1.mat','IR007_1.mat','OR007@6_1.mat','B007_1.mat'}; % ejemplo

fprintf('Procesando %d ficheros...\n', numel(fileList));

for k = 1:numel(fileList)
    fname = fileList{k};
    fpath = fullfile(rootDir, fname); 
    S = load(fpath);
    vnames = fieldnames(S);
    idxDE = find(contains(vnames,'DE') & contains(vnames,'time'), 1);
    x = double(S.(vnames{idxDE})(:));

    % Preprocesado
    x = detrend(x);
    x = x - mean(x);

    % Etiqueta
    label = startsWith(fname,'Normal') == 0;

    % Ventaneo
    Xw = buffer(x, Nwin, Nwin-hop, 'nodelay');   % col = ventana

    % ---- (1) X base (causal): A (tiempo) + C (envolvente)
    Xi = zeros(size(Xw,2), numel(feat_names_base));
    for w = 1:size(Xw,2)
        [A_time, ~, C_env, ~] = compute_features_window_blocks(Xw(:,w), Fs);
        Xi(w,:) = [A_time, C_env];
    end

    % ---- (2) X* privilegiado (global/no causal): B (freq) + D (TF)
    %      Calculado una sola vez por fichero, se replica para todas sus ventanas
    [~, B_freq_global, ~, D_tf_global] = compute_features_window_blocks(x, Fs, ...
        'global', true);   % <-- modo global para FFT/STFT
    xstar_file = [B_freq_global, D_tf_global];   % 1x15
    Xstar_i = repmat(xstar_file, size(Xw,2), 1);

    % Acumular
    X = [X; Xi];
    Xstar = [Xstar; Xstar_i];
    y = [y; repmat(label, size(Xi,1), 1)];
    file_ids = [file_ids; repmat(k, size(Xi,1), 1)];
    win_ids  = [win_ids; (1:size(Xi,1)).'];


end

fprintf('Hecho. X: %d x %d,  y: %d muestras (clases: %s)\n', size(X,1), size(X,2), numel(y), mat2str(unique(y)'));

% Barajar (opcional). OJO: si vas a hacer split por fichero, no barajes aquí.
idx = randperm(size(X,1));
X = X(idx,:); Xstar = Xstar(idx,:); y = y(idx);
file_ids = file_ids(idx); win_ids = win_ids(idx);

%Guardar 
save(fullfile(rootDir, 'cwru_12kDE_1hp_X_Xstar_y.mat'), ...
     'X','Xstar','y','feat_names_base','feat_names_star','file_ids','win_ids','Nwin','hop','Fs','fileList');


%% -------------------------------------------------------------
% FUNCIONES AUXILIARES
% -------------------------------------------------------------
function [A_time, B_freq, C_env, D_tf] = compute_features_window_blocks(x, Fs, varargin)
    % varargin: 'global', true -> fuerza modo global/no causal para B y D
    p = inputParser;
    addParameter(p,'global',false,@islogical);
    parse(p,varargin{:});
    doGlobal = p.Results.global;

    x = x(:); N = numel(x);

    % ---- A) Tiempo (12)
    rms_v   = sqrt(mean(x.^2));
    std_v   = std(x, 1);
    var_v   = var(x, 1);
    meanabs = mean(abs(x));
    p2p     = max(x) - min(x);
    sk      = skewness(x, 1);
    ku      = kurtosis(x, 1);
    peak    = max(abs(x)) + eps;
    crest   = peak / (rms_v + eps);
    impulse = peak / (meanabs + eps);
    shape   = rms_v / (meanabs + eps);
    margin  = peak / (meanabs^2 + eps);
    zc      = sum(x(1:end-1).*x(2:end) < 0) / N;
    A_time  = [rms_v, std_v, var_v, meanabs, p2p, sk, ku, crest, impulse, shape, margin, zc];

    % ---- B) Frecuencia (9)
    if doGlobal
        % Welch global con ventanas largas (no causal por construcción)
        [Pxx,F] = pwelch(x, hamming(4096,'periodic'), 2048, 8192, Fs);
    else
        Nfft = 1024;
        w = hamming(min(N, Nfft), 'periodic');
        xx = x(1:numel(w)).*w;
        Xf = fft(xx, Nfft);
        Pxx = (abs(Xf).^2)/sum(w.^2);
        Pxx = Pxx(1:floor(Nfft/2)+1);
        F   = linspace(0, Fs/2, numel(Pxx))';
    end
    Pn = Pxx / (sum(Pxx) + eps);
    fcent = sum(F .* Pn);
    fspread = sqrt( sum(((F - fcent).^2) .* Pn) );
    sentropy = -sum(Pn .* log2(Pn + eps)) / log2(numel(Pn));
    edges_khz = 0:1:6;
    bandE = zeros(numel(edges_khz)-1,1);
    for b = 1:numel(edges_khz)-1
        f1 = 1000*edges_khz(b); f2 = min(1000*edges_khz(b+1), Fs/2);
        idx = (F >= f1) & (F < f2);
        bandE(b) = log10(sum(Pxx(idx)) + eps);
    end
    B_freq = [fcent, fspread, sentropy, bandE(:).'];

    % ---- C) Envolvente (3) (igual en ambos modos: banda genérica)
    f1_env = 1000; f2_env = min(5000, 0.45*Fs);
    d = designfilt('bandpassiir','FilterOrder',6, ...
        'HalfPowerFrequency1',f1_env,'HalfPowerFrequency2',f2_env, ...
        'SampleRate',Fs);
    xf = filtfilt(d, x);
    env  = abs(hilbert(xf));
    rms_env   = sqrt(mean(env.^2));
    ku_env    = kurtosis(env, 1);
    crest_env = (max(env)+eps) / (rms_env + eps);
    C_env = [rms_env, ku_env, crest_env];

    % ---- D) Tiempo-frecuencia (6)
    if doGlobal
        % STFT densa, colapsada en el tiempo (no causal)
        [S,Fst,~] = spectrogram(x, hamming(1024,'periodic'), 512, 2048, Fs);
        Pstft = mean(abs(S).^2, 2);
    else
        [S,Fst,~] = spectrogram(x, hamming(256,'periodic'), 128, 512, Fs);
        Pstft = mean(abs(S).^2, 2);
    end
    stftE = zeros(numel(edges_khz)-1,1);
    for b = 1:numel(edges_khz)-1
        f1 = 1000*edges_khz(b); f2 = min(1000*edges_khz(b+1), Fs/2);
        idx = (Fst >= f1) & (Fst < f2);
        stftE(b) = log10(sum(Pstft(idx)) + eps);
    end
    D_tf = stftE(:).';
end

function names = feature_names_base()
    names = [ ...
        "rms","std","var","meanabs","p2p","skew","kurt", ...
        "crest","impulse","shape","margin","zcr", ...
        "env_rms","env_kurt","env_crest" ...
    ];
end

function names = feature_names_star()
    names = [ ...
        "f_centroid","f_spread","f_entropy", ...
        "fftE_0_1k","fftE_1_2k","fftE_2_3k","fftE_3_4k","fftE_4_5k","fftE_5_6k", ...
        "stftE_0_1k","stftE_1_2k","stftE_2_3k","stftE_3_4k","stftE_4_5k","stftE_5_6k" ...
    ];
end
