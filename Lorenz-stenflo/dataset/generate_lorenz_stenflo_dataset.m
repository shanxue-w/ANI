function generate_lorenz_stenflo_dataset()
    % ==== 系统参数 ====

    delta_t = 5e-2;
    steps_per_condition = 2;
    num_train_conditions = 1000;
    num_val_conditions = 200;

    total_train_samples = num_train_conditions * steps_per_condition;
    total_val_samples = num_val_conditions * steps_per_condition;

    % ==== 构建数据容器 ====
    train_inputs = zeros(total_train_samples, 5);   % [u1...u7, dt]
    train_outputs = zeros(total_train_samples, 4);  % 下一步 u1...u7

    val_inputs = zeros(total_val_samples, 5);
    val_outputs = zeros(total_val_samples, 4);

    opts = odeset('RelTol', 1e-10, 'AbsTol', 1e-12);
    tspan = 0:delta_t:(steps_per_condition * delta_t);

    % ==== 构建训练数据 ====
    idx = 1;
    for i = 1:num_train_conditions
        y0 = sample_initial_condition_lorenz_stenflo();
        [~, Y] = ode45(@(t,u) lorenz_stenflo_ode(t, u), tspan, y0, opts);
        for j = 1:steps_per_condition
            train_inputs(idx, :) = [Y(j, :), delta_t];
            train_outputs(idx, :) = Y(j+1, :);
            idx = idx + 1;
        end
    end

    % ==== 构建验证数据 ====
    idx = 1;
    for i = 1:num_val_conditions
        y0 = sample_initial_condition_lorenz_stenflo();
        [~, Y] = ode45(@(t,u) lorenz_stenflo_ode(t, u), tspan, y0, opts);
        for j = 1:steps_per_condition
            val_inputs(idx, :) = [Y(j, :), delta_t];
            val_outputs(idx, :) = Y(j+1, :);
            idx = idx + 1;
        end
    end

    % ==== 生成测试轨迹 ====
    num_trajectories = 4;
    total_time = 100;
    tspan_test = 0:delta_t:total_time;
    steps = length(tspan_test);
    test_trajectories = zeros(num_trajectories, steps, 4);
    for i = 1:num_trajectories
        y0 = sample_initial_condition_lorenz_stenflo();
        y0(1) = 2.0/3.0 * y0(1);
        y0(2) = 3.0/4.0 * y0(2);
        y0(4) = 1.0/2.0 * y0(4);
        [~, Y] = ode45(@(t,u) lorenz_stenflo_ode(t, u), tspan_test, y0, opts);
        test_trajectories(i, :, :) = Y;
    end

    % ==== 保存数据 ====
    save('lorenz_stenflo_train_data.mat', 'train_inputs', 'train_outputs');
    save('lorenz_stenflo_val_data.mat', 'val_inputs', 'val_outputs');
    save('lorenz_stenflo_test_trajectories.mat', 'test_trajectories');
end
