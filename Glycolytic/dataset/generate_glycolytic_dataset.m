function generate_glycolytic_dataset()
    % ==== 系统参数 ====
    p.J0 = 2.5; p.k1 = 100; p.k2 = 6; p.k3 = 16; p.k4 = 100;
    p.k5 = 1.28; p.k6 = 12; p.k = 1.8; p.kappa = 13;
    p.q = 4; p.K1 = 0.52; p.psi = 0.1; p.N = 1; p.A = 4;

    delta_t = 0.05;
    steps_per_condition = 4;
    num_train_conditions = 4000;
    num_val_conditions = 1000;

    total_train_samples = num_train_conditions * steps_per_condition;
    total_val_samples = num_val_conditions * steps_per_condition;

    % ==== 构建数据容器 ====
    train_inputs = zeros(total_train_samples, 8);   % [u1...u7, dt]
    train_outputs = zeros(total_train_samples, 7);  % 下一步 u1...u7

    val_inputs = zeros(total_val_samples, 8);
    val_outputs = zeros(total_val_samples, 7);

    opts = odeset('RelTol', 1e-10, 'AbsTol', 1e-12);
    tspan = 0:delta_t:(steps_per_condition * delta_t);

    % ==== 构建训练数据 ====
    idx = 1;
    for i = 1:num_train_conditions
        y0 = sample_initial_condition();
        [~, Y] = ode45(@(t,u) glycolytic_ode(t, u, p), tspan, y0, opts);
        for j = 1:steps_per_condition
            train_inputs(idx, :) = [Y(j, :), delta_t];
            train_outputs(idx, :) = Y(j+1, :);
            idx = idx + 1;
        end
    end

    % ==== 构建验证数据 ====
    idx = 1;
    for i = 1:num_val_conditions
        y0 = sample_initial_condition();
        [~, Y] = ode45(@(t,u) glycolytic_ode(t, u, p), tspan, y0, opts);
        for j = 1:steps_per_condition
            val_inputs(idx, :) = [Y(j, :), delta_t];
            val_outputs(idx, :) = Y(j+1, :);
            idx = idx + 1;
        end
    end

    % ==== 生成测试轨迹 ====
    num_trajectories = 200;
    total_time = 10;
    tspan_test = 0:5e-2:total_time;
    steps = length(tspan_test);
    test_trajectories = zeros(num_trajectories, steps, 7);

    for i = 1:num_trajectories
        if i == 1
            y0 = [0.2, 2.0, 0.054, 0.237, 0.152, 2.167, 0.07];
        else
            y0 = sample_initial_condition();
        end
        [~, Y] = ode45(@(t,u) glycolytic_ode(t, u, p), tspan_test, y0, opts);
        test_trajectories(i, :, :) = Y;
    end

    % ==== 保存数据 ====
    save('gly_train_data.mat', 'train_inputs', 'train_outputs');
    save('gly_val_data.mat', 'val_inputs', 'val_outputs');
    save('gly_test_trajectories_small.mat', 'test_trajectories');
end
