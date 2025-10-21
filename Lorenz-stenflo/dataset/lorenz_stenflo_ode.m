function dudt = lorenz_stenflo_ode(~, u)
    % 初始化导数向量
    dudt = zeros(4, 1);

    x = u(1);
    y = u(2);
    z = u(3);
    w = u(4);

    % 四个分量
    dudt(1) = (y-x)+1.5*w;
    dudt(2) = 26*x-x*z-y;
    dudt(3) = x*y-0.7*z;
    dudt(4) = -x-w;
end
