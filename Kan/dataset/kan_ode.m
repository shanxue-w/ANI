function dudt = kan_ode(~, u)
    % 初始化导数向量
    dudt = zeros(3, 1);

    xp = 0.4;
    yp = 2.009;
    xq = 0.08;
    yq = 2.876;
    K  = 0.98;
    N0 = 0.16129;
    P0 = 0.5;

    % 四个分量
    dudt(1) = u(1)*(1.-u(1)/K)-xp*yp*u(1)*u(2)/(u(1)+N0);
    dudt(2) = xp*u(2)*(yp*u(1)/(u(1)+N0) - 1.)-xq*yq*u(2)*u(3)/(u(2)+P0);
    dudt(3) = xq*u(3)*(yq*u(2)/(u(2)+P0) - 1.);
end
