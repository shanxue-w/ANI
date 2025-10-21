function generate_compound_dataset()

    delta_t = 0.05;
    total_samples = 7000;

    % ==== 构建监督学习数据 ====
    inputs = zeros(total_samples, 5);   % [u1...u7, dt]
    outputs = zeros(total_samples, 4);  % 下一步 u1...u7

    opts = odeset('RelTol', 1e-10, 'AbsTol', 1e-12);

    for i = 1:total_samples
        y0 = sample_initial_condition_compound();
        [~, Y] = ode45(@(t,u) compound_ode(t, u), [0 delta_t], y0, opts);
        inputs(i, :) = [y0, delta_t];
        outputs(i, :) = Y(end, :);
    end

    % 分割训练集 / 验证集
    train_inputs = inputs(1:5000, :);
    train_outputs = outputs(1:5000, :);

    val_inputs = inputs(5001:7000, :);
    val_outputs = outputs(5001:7000, :);

    % ==== 生成测试轨迹 ====
    num_trajectories = 3;
    total_time = 50.0;
    tspan = 0:delta_t:total_time;
    steps = length(tspan);

    test_trajectories = zeros(num_trajectories, steps, 4);

    for i = 1:num_trajectories
        y0 = sample_initial_condition_compound();
        [~, Y] = ode45(@(t,u) compound_ode(t, u), tspan, y0, opts);
        test_trajectories(i, :, :) = Y;
    end

    % ==== 保存数据 ====
    %save('compound_train_data.mat', 'train_inputs', 'train_outputs');
    %save('compound_val_data.mat', 'val_inputs', 'val_outputs');
    save('compound_test_trajectories1.mat', 'test_trajectories');
end
