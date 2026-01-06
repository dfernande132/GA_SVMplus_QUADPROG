
function plot_cwru_signals_en()
% -------------------------------------------------------------
% CWRU (12k Drive End) plotting helper
%   Figures:
%     (1) Time-domain waveforms (first maxSecondsFig1 seconds)
%     (2) Windowed RMS profiles (enforce maxSecondsFig2 seconds)
%     (3) Power Spectral Density (Welch) (maxSecondsFig3 seconds)
%     (4) Time–frequency spectrogram (STFT, dB) (maxSecondsFig4 seconds)
% -------------------------------------------------------------

    % --- Config
    rootDir = 'CWRU_Dataset\';  % <--- adjust if needed
    keepForPlot = {'Normal_1.mat','IR007_1.mat','OR007@6_1.mat','B007_1.mat'};
    Fs   = 12000;        % 12 kHz (Drive End)
    Nwin = 2048;
    hop  = Nwin/2;

    maxSecondsFig1 = 1;   % seconds shown in Figure 1 (time waveforms)
    maxSecondsFig2 = 10;  % seconds enforced in Figure 2 (RMS)
    maxSecondsFig3 = 10;  % seconds used in Figure 3 (PSD Welch)
    maxSecondsFig4 = 10;  % seconds used in Figure 4 (spectrogram)

    % STFT params for spectrogram (Fig. 4)
    stft_win   = 1024;
    stft_ovlp  = 512;
    stft_nfft  = 2048;
    fmax_khz   = 6;       % show up to 6 kHz

    % --- Load requested signals
    signals = struct(); order = {};
    for i = 1:numel(keepForPlot)
        fname = keepForPlot{i};
        fpath = fullfile(rootDir, fname);
        if ~exist(fpath, 'file')
            warning('File not found: %s', fpath);
            continue;
        end
        try
            [x, usedField] = load_DE_time(fpath);
            x = detrend(double(x(:)));
            x = x - mean(x);
            key = matlab.lang.makeValidName(fname);
            signals.(key).x = x;
            signals.(key).fname = fname;
            signals.(key).field = usedField;
            order{end+1} = key; %#ok<AGROW>
        catch ME
            warning('Error reading %s: %s', fpath, ME.message);
        end
    end

    if isempty(fieldnames(signals))
        error('No signals loaded. Check rootDir and keepForPlot names.');
    end

    %% -------- Figure 1: time-domain waveforms (first N seconds)
    figure('Name','Time-domain waveforms','Color','w');
    tiledlayout(2,2,'Padding','compact','TileSpacing','compact');
    for i = 1:min(4, numel(order))
        fld = order{i};
        x   = signals.(fld).x;
        Ns  = min(numel(x), round(Fs*maxSecondsFig1));
        t   = (0:Ns-1)/Fs;

        nexttile;
        plot(t, x(1:Ns), 'LineWidth', 1);
        grid on;
        title(strrep(signals.(fld).fname, '_','\_'));
        xlabel('Time (s)'); ylabel('Amplitude (a.u.)');
        subtitle(sprintf('Channel: %s', signals.(fld).field), 'Interpreter','none');
    end

    %% -------- Figure 2: windowed RMS profiles (enforce fixed seconds)
    figure('Name','Windowed RMS profiles','Color','w');
    tiledlayout('vertical','Padding','compact','TileSpacing','compact');
    for i = 1:numel(order)
        fld = order{i};
        x   = signals.(fld).x;

        Ns_max = min(numel(x), round(Fs*maxSecondsFig2));
        x_plot = x(1:Ns_max);

        Xw = buffer(x_plot, Nwin, Nwin-hop, 'nodelay');  % cols = windows
        r  = sqrt(mean(Xw.^2, 1));
        tt = ((0:numel(r)-1) * (hop/Fs));               % window start times

        nexttile;
        stem(tt, r, 'filled'); grid on;
        title(['Windowed RMS - ', strrep(signals.(fld).fname,'_','\_')]);
        xlabel('Time (s)'); ylabel('RMS (a.u.)');
    end

    %% -------- Figure 3: Power Spectral Density (Welch)
    figure('Name','Power Spectral Density (Welch)','Color','w');
    tiledlayout('vertical','Padding','compact','TileSpacing','compact');
    for i = 1:numel(order)
        fld = order{i};
        x   = signals.(fld).x;
        Ns_max = min(numel(x), round(Fs*maxSecondsFig3));
        x_plot = x(1:Ns_max);

        win  = hamming(4096,'periodic');
        nover= 2048;
        nfft = 8192;
        [Pxx, F] = pwelch(x_plot, win, nover, nfft, Fs);

        nexttile;
        plot(F, 10*log10(Pxx), 'LineWidth', 1); grid on;
        xlim([0, Fs/2]);
        title(['PSD (Welch) - ', strrep(signals.(fld).fname,'_','\_')]);
        xlabel('Frequency (Hz)'); ylabel('PSD (dB/Hz)');
    end

    %% -------- Figure 4: Time–frequency spectrogram (STFT, dB)
    figure('Name','Time–frequency (spectrogram)','Color','w');
    tiledlayout('vertical','Padding','compact','TileSpacing','compact');
    for i = 1:numel(order)
        fld = order{i};
        x   = signals.(fld).x;
        Ns_max = min(numel(x), round(Fs*maxSecondsFig4));
        x_plot = x(1:Ns_max);

        % STFT
        [S, F, T] = spectrogram(x_plot, hamming(stft_win,'periodic'), ...
                                stft_ovlp, stft_nfft, Fs);
        P = abs(S).^2;          % power
        PdB = 10*log10(P + eps);

        % Dynamic range clipping for nicer visuals
        pr = prctile(PdB(:), [1 99]);
        PdB = max(min(PdB, pr(2)), pr(1));

        % Show up to fmax_khz kHz
        fmask = F <= fmax_khz*1000;
        Fk = F(fmask)/1000;     % kHz
        PdBk = PdB(fmask, :);

        nexttile;
        imagesc(T, Fk, PdBk); axis xy;
        colormap(get_colormap());
        colorbar;
        title(['Spectrogram - ', strrep(signals.(fld).fname,'_','\_')]);
        xlabel('Time (s)'); ylabel('Frequency (kHz)');
    end

    fprintf('Done. Fs=%d Hz, Nwin=%d, hop=%d. Plotted signals: %d\n', Fs, Nwin, hop, numel(order));
end

function [x, usedField] = load_DE_time(fpath)
% Loads a CWRU .mat and returns an accelerometer channel:
%   Priority: DE_time -> DE_time_* -> FE_time -> any *_time

    S = load(fpath);
    v = fieldnames(S);

    % Priority 1: DE_time exact
    idx = find(strcmpi(v, 'DE_time'), 1);
    if ~isempty(idx)
        usedField = v{idx};
        x = S.(usedField);
        return;
    end

    % Priority 2: any DE* with *time*
    idx = find(contains(v, 'DE') & contains(v,'time'), 1);
    if ~isempty(idx)
        usedField = v{idx};
        x = S.(usedField);
        return;
    end

    % Priority 3: FE_time
    idx = find(strcmpi(v, 'FE_time'), 1);
    if ~isempty(idx)
        usedField = v{idx};
        x = S.(usedField);
        return;
    end

    % Priority 4: any *_time
    idx = find(endsWith(v, '_time'), 1);
    if ~isempty(idx)
        usedField = v{idx};
        x = S.(usedField);
        return;
    end

    error('No *_time channel found in %s', fpath);
end

function cm = get_colormap()
% Use turbo if available (R2020b+), else fall back to parula
    if exist('turbo', 'file') == 2
        cm = turbo;
    else
        cm = parula;
    end
end
