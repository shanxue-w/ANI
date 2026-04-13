% metric_trajectories_for_chaos.m

here = fileparts(mfilename('fullpath'));
matfile = fullfile(here, 'trajectories_for_chaos.mat');

if exist(matfile, 'file') ~= 2
    error('Missing %s', matfile);
end

S = load(matfile);
dt = double(S.dt(1));
fs = 1 / dt;

% Order: row 1 = reference truth; rows 2–3 = models (vs your true/base/4th; no 2nd in this .mat)
vars = {'truth', 'rk4_prior_plus_B_baseline', 'ani4_strang_B_ani4'};

nVar = numel(vars);
nTraj = size(S.truth, 1);
T = size(S.truth, 2);

K = 2000;
K = min(K, T);
if K < 500
    warning('metric_trajectories_for_chaos: T=%d < 500, using K=%d.', T, K);
end

% 1-based trajectory indices (default: all trajectories in the file)
traj_indices = 1:nTraj;

% Optional: use only (x,y,z) for metrics [1 2 3]; [] = all 4 states
state_cols = [];

get_metrics = @(d) [correlationDimension(d), ...
                    approximateEntropy(d), ...
                    lyapunovExponent(d, fs)];

all_metrics = zeros(nVar, 3, numel(traj_indices));

for idx = 1:numel(traj_indices)
    t_idx = traj_indices(idx);
    for i = 1:nVar
        data = squeeze(S.(vars{i})(t_idx, 1:K, :));
        if ~isempty(state_cols)
            data = data(:, state_cols);
        end
        if size(data, 1) ~= K
            data = reshape(data, K, []);
        end
        all_metrics(i, :, idx) = get_metrics(data);
    end
end

avg_metrics = mean(all_metrics, 3);

m_true = avg_metrics(1, :);
den = abs(m_true);
den(den < eps) = NaN;
rel_errors = abs(avg_metrics(2:end, :) - m_true) ./ den;

methods = {'Baseline (RK4+SINDy)', 'ANI4 (Strang)'};
metrics_names = {'Corr Dimension', 'App Entropy', 'Lyapunov Exp'};

fprintf('\n%s\n', matfile);
fprintf('fs = 1/dt = %.6g (dt = %.6g)\n', fs, dt);
fprintf('K = %d time steps, nTraj averaged = %d\n\n', K, numel(traj_indices));

fprintf('%-22s | %-12s | %-12s | %-12s\n', 'Metric', 'Ground Truth', 'Method Value', 'Rel. Error');
fprintf('------------------------------------------------------------------------------\n');

for m = 1:3
    fprintf('--- %s ---\n', metrics_names{m});
    for i = 1:size(rel_errors, 1)
        val = avg_metrics(i + 1, m);
        err = rel_errors(i, m) * 100;
        fprintf('%-22s | %-12.5f | %-12.5f | %-7.2f%%\n', ...
                methods{i}, m_true(m), val, err);
    end
    fprintf('------------------------------------------------------------------------------\n');
end
