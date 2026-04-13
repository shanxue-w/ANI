% 模拟参数
K = 500; fs = 1/5e-2;
files = {'true', 'base', '2th', '4th'}; 
fields = {'true_traj', 'NN_traj', 'NN_traj', 'NN_traj'};
traj_indices = 0:3;


all_metrics = zeros(4, 3, length(traj_indices));

get_metrics = @(d) [correlationDimension(d), ...
                    approximateEntropy(d), ...
                    lyapunovExponent(d, fs)];

for idx = 1:length(traj_indices)
    t_idx = traj_indices(idx);
    for i = 1:4
        fname = sprintf('ANI_Lorenz_stenflo_%s_%d_h32.mat', files{i}, t_idx);
        if exist(fname, 'file')
            tmp = load(fname);
            data = tmp.(fields{i});
            all_metrics(i, :, idx) = get_metrics(data(1:K, :));
        end
    end
end


avg_metrics = mean(all_metrics, 3);


m_true = avg_metrics(1, :);
rel_errors = abs(avg_metrics(2:4, :) - m_true) ./ abs(m_true);


methods = {'Baseline', 'ANI-2nd', 'ANI-4th'};
metrics_names = {'Corr Dimension', 'App Entropy', 'Lyapunov Exp'};

fprintf('\n%-18s | %-12s | %-12s | %-12s\n', 'Metric', 'Ground Truth', 'Method Value', 'Rel. Error');
fprintf('--------------------------------------------------------------------------\n');

for m = 1:3
    fprintf('--- %s ---\n', metrics_names{m});
    for i = 1:3 
        val = avg_metrics(i+1, m);
        err = rel_errors(i, m) * 100;
        fprintf('%-18s | %-12.5f | %-12.5f | %-7.2f%%\n', ...
                methods{i}, m_true(m), val, err);
    end
    fprintf('--------------------------------------------------------------------------\n');
end



avg_metrics = mean(all_metrics, 3);
std_metrics = std(all_metrics, 0, 3);
sem_metrics = std_metrics / sqrt(size(all_metrics, 3));


methods_display = {'Ground Truth', 'Baseline', 'ANI-2nd', 'ANI-4th'};
metrics_names = {'Corr Dimension', 'App Entropy', 'Lyapunov Exp'};

fprintf('\n%-18s | %-20s | %-10s\n', 'Metric', 'Value (Mean ± SEM)', 'Rel. Error');
fprintf('--------------------------------------------------------------------------\n');

for m = 1:3
    fprintf('--- %s ---\n', metrics_names{m});
    

    fprintf('%-18s | %.2f ± %.2f        | -\n', ...
            methods_display{1}, avg_metrics(1, m), sem_metrics(1, m));
        

    for i = 1:3 
        val = avg_metrics(i+1, m);
        err_val = sem_metrics(i+1, m);
        rel_err = abs(val - avg_metrics(1, m)) / abs(avg_metrics(1, m)) * 100;
        

        fprintf('%-18s | %.2f ± %.2f        | %-7.1f%%\n', ...
                methods{i}, val, err_val, rel_err);
    end
    fprintf('--------------------------------------------------------------------------\n');
end