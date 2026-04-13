% Lorenz 系统参数
sigma = 1;
rho = 26;
beta = 0.7;

% Lorenz 系统定义
lorenz = @(t, u) [
    sigma * (u(2) - u(1));
    rho * u(1) - u(2) - u(1) * u(3);
    u(1) * u(2) - beta * u(3)
];

% ======== 设置初始条件 ==========
u0 = [13.877355519295373; -13.172332152300159; 16.073708997653998];  % <<< 可替换成你自己的初始值

% ======== 时间设置 ===============
dt = 1e-2;
T_final = 100;
tspan = 0:dt:T_final;

% ======== 精度设置 ===============
options = odeset('RelTol', 1e-10, 'AbsTol', 1e-12);

% ======== 解算 ===================
[t1, u1] = ode45(lorenz, tspan, u0, options);

% ======== 保存 ===================
save('lorenz_tol1e10.mat', 't1', 'u1', 'sigma', 'rho', 'beta', 'u0', 'dt')

% ======== 可视化（可选） ==========
figure;
subplot(1,1,1)
plot3(u1(:,1), u1(:,2), u1(:,3), 'b')
title('Lorenz Trajectory (tol=1e-10)')
xlabel('x'); ylabel('y'); zlabel('z'); grid on
