function dudt = morris_lecar_ode(~, u)
    % Extract state variables
    V = u(1);
    w = u(2);

    % --- Morris-Lecar System Parameters (from Table 1) ---
    % Type I parameters are used as specified in the table.
    C = 20;     % Membrane capacitance
    gL = 2;     % Leak conductance
    VL = -60;   % Leak reversal potential
    gCa = 4;    % Calcium conductance
    VCa = 120;  % Calcium reversal potential
    gK = 8;     % Potassium conductance
    VK = -84;   % Potassium reversal potential
    v1 = -1.2;  % Half-activation voltage for m_infinity
    v2 = 18;    % Slope for m_infinity
    v3 = 12;    % Half-activation voltage for w_infinity
    v4 = 17.4;  % Slope for w_infinity
    phi = 0.066;% Activation time constant factor

    % --- Define I_app (Applied Current) ---
    % The provided snippet does not define I_app.
    % For a complete simulation, I_app must be defined.
    % For demonstration, we'll set it to a constant value, but it can be time-varying.
    I_app = 60;

    % --- Calculate auxiliary variables ---

    % Equilibrium open fraction for Ca2+ current (m_infinity)
    % m_infinity = 0.5 * [1 + tanh(V - v1)] / v2
    m_infinity = 0.5 * (1 + tanh((V - v1) / v2));

    % Equilibrium open fraction for K+ current (w_infinity)
    % w_infinity = 0.5 * [1 + tanh((V - v3))] / v4
    w_infinity = 0.5 * (1 + tanh((V - v3) / v4));

    % Activation time constant for the delayed rectifier (tau)
    % tau = 1 / (cosh(V - v3) * (2 * v4))
    tau = 1 /cosh((V - v3)/(2*v4));

    % --- Calculate the derivatives ---

    % dV/dt equation (Equation 1, first line)
    % dV/dt = (-gCa * (V - VCa) * m_infinity - gK * (V - VK) * w - gL * (V - VL) + I_app) / C
    dVdt = (-gCa * (V - VCa) * m_infinity - gK * (V - VK) * w - gL * (V - VL) + I_app) / C;

    % dw/dt equation (Equation 1, second line)
    % dw/dt = phi * (w_infinity - w) / tau
    dwdt = phi * (w_infinity - w) / tau;

    % Assemble the output vector dudt
    dudt = [dVdt; dwdt];

end