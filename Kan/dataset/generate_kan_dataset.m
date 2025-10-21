function generate_kan_dataset()
    % ==== 系统参数 ====
    opts = odeset('RelTol', 1e-10, 'AbsTol', 1e-12);
    delta_t = 1e-1;
 
    steps_per_condition = 2;
    num_train_conditions = 10000;
    num_val_conditions = 3000;
    

    total_train_samples = num_train_conditions * steps_per_condition;
    total_val_samples = num_val_conditions * steps_per_condition;

    % ==== 构建数据容器 ====
    train_inputs = zeros(total_train_samples, 4);   % [u1...u7, dt]
    train_outputs = zeros(total_train_samples, 3);  % 下一步 u1...u7

    val_inputs = zeros(total_val_samples, 4);
    val_outputs = zeros(total_val_samples, 3);

    
    tspan = 0:delta_t:(steps_per_condition * delta_t);

    
    % ==== 构建训练数据 ====
    idx = 1;
    for i = 1:num_train_conditions
        y0 = sample_initial_condition_kan();
        [~, Y] = ode45(@(t,u) kan_ode(t, u), tspan, y0, opts);
        for j = 1:steps_per_condition
            train_inputs(idx, :) = [Y(j, :), delta_t];
            train_outputs(idx, :) = Y(j+1, :);
            idx = idx + 1;
        end
    end
    % ==== 构建验证数据 ====
    idx = 1;
    for i = 1:num_val_conditions
        y0 = sample_initial_condition_kan();
        [~, Y] = ode45(@(t,u) kan_ode(t, u), tspan, y0, opts);
        for j = 1:steps_per_condition
            val_inputs(idx, :) = [Y(j, :), delta_t];
            val_outputs(idx, :) = Y(j+1, :);
            idx = idx + 1;
        end
    end

    % ==== 生成测试轨迹 ====
    num_trajectories = 10;
    total_time = 200;
    tspan_test = 0:delta_t:total_time;
    steps = length(tspan_test);
    test_trajectories = zeros(num_trajectories, steps, 3);
    for i = 1:num_trajectories
        disp(i)
        if i == 1
            y0 = [1. 0.3 0.68];
        elseif i == 2
            y0 = [0.6250 0.0278 0.0548];
        elseif i== 3
            y0 = [0.5105 0.5985 0.2815];
        else
            y0 = sample_initial_condition_kan();
        end
        [~, Y] = ode45(@(t,u) kan_ode(t, u), tspan_test, y0, opts);
        test_trajectories(i, :, :) = Y;
    end

    % ==== 保存数据 ====
    %save('kan_train_data.mat', 'train_inputs', 'train_outputs');
    %save('kan_val_data.mat', 'val_inputs', 'val_outputs');
    save('kan_test_trajectories_1.mat', 'test_trajectories');
end
