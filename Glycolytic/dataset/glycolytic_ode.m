function dudt = glycolytic_ode(~, u, p)
    dudt = zeros(7, 1);
    denom = 1 + (u(6)/p.K1)^p.q;

    dudt(1) = p.J0 - p.k1*u(1)*u(6)/denom;
    dudt(2) = 2*p.k1*u(1)*u(6)/denom - p.k2*u(2)*(p.N - u(5)) - p.k6*u(2)*u(5);
    dudt(3) = p.k2*u(2)*(p.N - u(5)) - p.k3*u(3)*(p.A - u(6));
    dudt(4) = p.k3*u(3)*(p.A - u(6)) - p.k4*u(4)*u(5) - p.kappa*(u(4) - u(7));
    dudt(5) = p.k2*u(2)*(p.N - u(5)) - p.k4*u(4)*u(5) - p.k6*u(2)*u(5);
    dudt(6) = -2*p.k1*u(1)*u(6)/denom + 2*p.k3*u(3)*(p.A - u(6)) - p.k5*u(6);
    dudt(7) = p.psi*p.kappa*(u(4) - u(7)) - p.k*u(7);
end
