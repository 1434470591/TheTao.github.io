function fresh_pilot_aided_demo()
    % =========================================================================
    % 基于导频 (Pilot-Aided) 的 FRESH 滤波仿真
    % 场景：利用头部已知的短导频序列计算权重，应用于后续未知负载数据
    % =========================================================================

    % 1. 仿真参数
    % -------------------------------------------------------------------------
    Fs = 1000;              % 采样率
    Rb_soi = 50;            % 目标信号波特率
    fc_soi = 100;           % 目标载波
    
    Rb_snoi = 65;           % 干扰波特率 (必须与目标不同)
    fc_snoi = 100;          % 干扰载波 (同频干扰!)
    
    N_pilot_syms = 100;     % 导频符号数 (已知)
    N_payload_syms = 2000;  % 负载符号数 (未知)
    
    SNR_dB = 20;
    SIR_dB = -5;            % 干扰比信号强 5dB
    
    fprintf('=== 导频辅助 FRESH 滤波 ===\n');
    fprintf('导频长度: %d 符号, 负载长度: %d 符号\n', N_pilot_syms, N_payload_syms);
    fprintf('信干比 (SIR): %.1f dB (强干扰)\n', SIR_dB);

    % 2. 信号生成
    % -------------------------------------------------------------------------
    % A. 生成目标信号 (包含 Pilot + Payload)
    [sig_soi, pilot_ref, payload_ref, time_vec, split_idx] = ...
        gen_packet_signal(N_pilot_syms, N_payload_syms, Rb_soi, fc_soi, Fs);
    
    % B. 生成干扰信号 (连续的随机干扰)
    % 干扰长度要覆盖整个目标数据包
    total_len = length(sig_soi);
    t = time_vec;
    snoi_params.Rb = Rb_snoi;
    snoi_params.fc = fc_snoi;
    snoi_params.power = 10^(-SIR_dB/10); % 根据SIR调整功率
    sig_snoi = gen_continuous_interference(t, snoi_params);
    
    % C. 混合接收
    noise = (randn(size(sig_soi)) + 1j*randn(size(sig_soi)))/sqrt(2) * 10^(-SNR_dB/20);
    x = sig_soi + sig_snoi + noise;
    
    % 3. 构建 FRESH 观测矩阵 (对整个数据包)
    % -------------------------------------------------------------------------
    % 定义循环频率: 基带 + 波特率频移 (利用循环平稳性)
    alpha_shifts = [0, Rb_soi, -Rb_soi]; 
    L_taps = 5; % 滤波器抽头
    
    fprintf('构建数据矩阵 Z (使用循环频率: %s)...\n', num2str(alpha_shifts));
    
    Z_all = [];
    for k = 1:length(alpha_shifts)
        alpha = alpha_shifts(k);
        % 频移
        x_shifted = x .* exp(1j * 2 * pi * alpha * t);
        % 构造 TDL
        Z_branch = make_tdl(x_shifted, L_taps);
        Z_all = [Z_all, Z_branch];
    end
    
    % 对齐数据 (去除 TDL 边缘)
    % 有效数据从 L_taps 开始
    valid_idx = L_taps:length(x);
    Z_valid = Z_all(valid_idx, :);
    t_valid = t(valid_idx);
    
    % 4. 训练阶段 (仅使用导频部分)
    % -------------------------------------------------------------------------
    % 确定导频在 Z_valid 中的结束位置
    % 修正点：计算长度时需要 +1，因为 MATLAB 切片是闭区间 [Start, End]
    train_len = split_idx - L_taps + 1; 
    
    if train_len <= 0
        error('导频太短，不足以填充滤波器抽头！');
    end
    
    % 提取训练数据 (X_train) 和 训练标签 (d_train)
    % Z_valid 的第 1 行对应原信号的第 L_taps 个样点
    Z_train = Z_valid(1:train_len, :);
    
    % d_train 必须是纯净的导频波形
    % 取出的区间必须与 Z_train 对应的时域区间严格一致
    d_train = pilot_ref(L_taps : split_idx); 
    
    % 维度检查 (调试用)
    fprintf('开始训练滤波器...\n');
    fprintf('Z_train 维度: [%d x %d]\n', size(Z_train, 1), size(Z_train, 2));
    fprintf('d_train 维度: [%d x %d]\n', size(d_train, 1), size(d_train, 2));
    
    if size(Z_train, 1) ~= size(d_train, 1)
        error('维度依然不匹配！请检查 train_len 计算');
    end

    % 计算最佳权重 (最小二乘解: w = Z \ d)
    % 增加正则化项
    R_train = (Z_train' * Z_train) + 1e-6*eye(size(Z_train,2));
    p_train = Z_train' * d_train;
    w_opt = R_train \ p_train;
    
    % 5. 应用阶段 (Payload Data)
    % -------------------------------------------------------------------------
    % 将训练好的权重应用到剩余的未知数据 (Payload)
    Z_payload = Z_valid(train_len+1:end, :);
    d_payload_ideal = payload_ref(1:size(Z_payload,1)); % 仅用于最后计算误码率，不参与滤波
    
    y_payload = Z_payload * w_opt;
    
    % 6. 结果评估与绘图
    % -------------------------------------------------------------------------
    % 计算 NMSE
    nmse_db = 10*log10(mean(abs(y_payload - d_payload_ideal).^2) / mean(abs(d_payload_ideal).^2));
    fprintf('负载数据恢复 NMSE: %.2f dB\n', nmse_db);
    
    figure('Color', 'w', 'Position', [100, 100, 1000, 600]);
    
    subplot(2,2,1);
    plot_spectrum(sig_soi, Fs); title('期望信号频谱 (强干扰)');
    
    subplot(2,2,2);
    plot_spectrum(sig_snoi, Fs); title('干扰信号频谱 (强干扰)');

    subplot(2,2,3);
    plot_spectrum(x, Fs); title('接收信号频谱 (强干扰)');
    
    subplot(2,2,4);
    plot_spectrum(y_payload, Fs); title('FRESH 滤波输出频谱');
    
    % subplot(2,2,3);
    % % 画最后 500 个点的星座图，确保是 Payload 部分
    % plot(x(end-1000:end), '.'); axis square; grid on;
    % title('滤波前星座图 (Payload)'); xlim([-4 4]); ylim([-4 4]);
    % 
    % subplot(2,2,4);
    % plot(y_payload(end-1000:end), '.'); axis square; grid on;
    % title(['滤波后星座图 (Payload), NMSE=' num2str(nmse_db,'%.1f') 'dB']);
    % xlim([-2 2]); ylim([-2 2]);
    
end

% =========================================================================
% 辅助函数
% =========================================================================

function [sig_all, pilot_part, payload_part, t, split_idx] = ...
    gen_packet_signal(n_pilot, n_payload, Rb, fc, Fs)
    
    sps = round(Fs / Rb);
    
    % 1. 生成比特序列
    % 导频: 固定序列 (例如全1或者伪随机但已知)
    rng(42); % 固定种子保证导频已知
    bits_pilot = 2*randi([0,1], n_pilot, 1) - 1; 
    
    % 负载: 随机未知数据
    rng('shuffle');
    bits_payload = 2*randi([0,1], n_payload, 1) - 1;
    
    bits_all = [bits_pilot; bits_payload];
    
    % 2. 脉冲成型 (一次性完成，保证波形连续)
    upsampled = zeros(length(bits_all)*sps, 1);
    upsampled(1:sps:end) = bits_all;
    
    h = manual_srrc(0.5, 6, sps);
    sig_base = conv(upsampled, h, 'same');
    
    % 3. 生成时间向量
    N_total = length(sig_base);
    t = (0:N_total-1)'/Fs;
    
    % 4. 调制
    sig_all = sig_base .* exp(1j * 2 * pi * fc * t);
    
    % 5. 分割参考信号 (用于训练和验证)
    % 计算导频结束的采样点索引
    split_idx = n_pilot * sps;
    
    pilot_part = sig_all(1:split_idx);
    payload_part = sig_all(split_idx+1:end);
end

function sig = gen_continuous_interference(t, p)
    % 生成与目标信号长度一致的干扰
    Fs = 1/(t(2)-t(1));
    sps = round(Fs / p.Rb);
    
    num_syms = ceil(length(t)/sps);
    bits = 2*randi([0,1], num_syms, 1) - 1;
    
    upsampled = zeros(length(bits)*sps, 1);
    upsampled(1:sps:end) = bits;
    
    h = manual_srrc(0.5, 6, sps);
    sig_base = conv(upsampled, h, 'same');
    
    % 截断/补零到 t 的长度
    if length(sig_base) > length(t)
        sig_base = sig_base(1:length(t));
    else
        sig_base = [sig_base; zeros(length(t)-length(sig_base), 1)];
    end
    
    sig = sqrt(p.power) * sig_base .* exp(1j * 2 * pi * p.fc * t);
end

function M = make_tdl(x, L)
    N = length(x);
    M = zeros(N, L);
    x_pad = [zeros(L-1,1); x];
    for i = 1:L
        M(:, i) = x_pad(L-i+1 : end-i+1);
    end
end

function h = manual_srrc(beta, span, sps)
    t = (-span/2 : 1/sps : span/2) + 1e-8;
    num = sin(pi*t*(1-beta)) + 4*beta*t.*cos(pi*t*(1+beta));
    den = (pi*t) .* (1 - (4*beta*t).^2);
    h = num ./ den;
    h = h / sqrt(sum(h.^2));
end

function plot_spectrum(x, Fs)
    L = length(x);
    nfft = 2^nextpow2(L);
    Y = fft(x, nfft);
    f = Fs*(-nfft/2:nfft/2-1)/nfft;
    P = fftshift(abs(Y).^2/L);
    plot(f, 10*log10(P+1e-12)); 
    grid on; xlabel('Hz'); ylabel('dB');
    xlim([0 Fs/2]);
end