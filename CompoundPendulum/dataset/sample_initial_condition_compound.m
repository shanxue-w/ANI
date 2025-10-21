function u0 = sample_initial_condition_compound()
    ranges = [-pi/2 pi/2;
              -pi pi;
              -pi/2 pi/2;
              -pi pi];
    u0 = zeros(1,4);
    for i = 1:4
        u0(i) = rand() * (ranges(i,2) - ranges(i,1)) + ranges(i,1);
    end
end
