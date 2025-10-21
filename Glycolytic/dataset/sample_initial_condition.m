function u0 = sample_initial_condition()
    % 定义各变量的采样范围（见文中 domain D）
    ranges = [0.15 1.6;
              0.19 2.16;
              0.04 0.2;
              0.1 0.35;
              0.08 0.3;
              0.14 2.67;
              0.05 0.1];
    u0 = zeros(1,7);
    for i = 1:7
        u0(i) = rand() * (ranges(i,2) - ranges(i,1)) + ranges(i,1);
    end
end
