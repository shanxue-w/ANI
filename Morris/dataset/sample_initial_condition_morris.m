function u0 = sample_initial_condition_morris()
    ranges = [-60 60;
              0 1];
    u0 = zeros(1,2);
    for i = 1:2
        u0(i) = rand() * (ranges(i,2) - ranges(i,1)) + ranges(i,1);
    end
end
