function u0 = sample_initial_condition_lorenz_stenflo()
    ranges = [-30 30;
              -40 40;
              -0  40;
              -20 20];
    u0 = zeros(1,4);
    for i = 1:4
        u0(i) = rand() * (ranges(i,2) - ranges(i,1)) + ranges(i,1);
    end
end
