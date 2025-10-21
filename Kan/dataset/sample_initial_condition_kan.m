function u0 = sample_initial_condition_kan()
    ranges = [0. 1.;
              0. 1.;
              0. 1.];
    u0 = zeros(1,3);
    for i = 1:3
        u0(i) = rand() * (ranges(i,2) - ranges(i,1)) + ranges(i,1);
    end
end
