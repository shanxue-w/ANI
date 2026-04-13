clear; clc;


max_K = 500; 
step_K = 10;
K_values = 50:step_K:max_K; 
fs = 1/5e-2;

files = {'true', 'base', '2th', '4th'}; 
fields = {'true_traj', 'NN_traj', 'NN_traj', 'NN_traj'};
methods_map = {'Reference', 'Baseline', 'ANI-2', 'ANI-4'};
traj_indices = 0:3; %


get_corr = @(d) correlationDimension(d);
get_app = @(d) approximateEntropy(d);
get_lya = @(d) lyapunovExponent(d, fs);

filename = 'chaos_metrics_data.csv';
fid = fopen(filename, 'w');

fprintf(fid, 'K,Trajectory,Method,Metric,Value\n');




for t_idx = 1:length(traj_indices)
    traj_id = traj_indices(t_idx);
    fprintf('Trajectory %d...\n', traj_id);
    

    RawData = cell(1, 4);
    for m = 1:4
        fname = sprintf('ANI_Lorenz_stenflo_%s_%d.mat', files{m}, traj_id);
        if exist(fname, 'file')
            tmp = load(fname);
            RawData{m} = tmp.(fields{m});
        end
    end
    

    for k = K_values
        for m = 1:4
            data_full = RawData{m};
            method_name = methods_map{m};
            
            if ~isempty(data_full) && size(data_full, 1) >= k
                sub_data = data_full(1:k, :);
                

                try
                    val_corr = get_corr(sub_data);
                    val_app  = get_app(sub_data);
                    val_lya  = get_lya(sub_data);
                    

                    fprintf(fid, '%d,%d,%s,%s,%.6f\n', k, traj_id, method_name, 'Correlation Dimension', val_corr);
                    fprintf(fid, '%d,%d,%s,%s,%.6f\n', k, traj_id, method_name, 'Approximate Entropy', val_app);
                    fprintf(fid, '%d,%d,%s,%s,%.6f\n', k, traj_id, method_name, 'Lyapunov Exponent', val_lya);
                catch

                end
            end
        end
    end
end

fclose(fid);
fprintf(' %s\n', filename);