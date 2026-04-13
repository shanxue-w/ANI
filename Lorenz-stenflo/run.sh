#!/usr/bin/env bash

# We already have data so this is not needed.
# cd dataset && matlab -batch generate_lorenz_stenflo_dataset && python data.py

bash run_hidden_variants.sh

for dir in ./ dataset/; do
    cp NeuralRK4_h32/*.mat 2th_h32/*.mat 4th_h32/*.mat "$dir"
done

python plot.py

python learn_missing_physics_compare.py
python symbolic_refinement_rollout.py
python reproduce_symbolic_writeback.py

# For matlab 
matlab -batch "metric"
matlab -batch "metric_adaptive.m"

python plot_time.py