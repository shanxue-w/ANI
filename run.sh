#!/usr/bin/env bash
set -ex

pip install -e .

cd Pendulum
python -u plot.py
cd ..

cd Lorenz-stenflo
python -u plot.py
python -u symbolic_refinement_rollout.py
python -u reproduce_symbolic_writeback.py
cd ..

cd Glycolytic
python -u plot.py
cd ..

cd CompoundPendulum
python -u plot.py
cd ..

cd Morris
python -u plot.py
cd ..

cd Kan
python -u plot.py
cd ..

cd Euler
python -u plot_sod.py
python -u plot_1.py
python -u plot_2.py
cd ..

cd Fitzhugh-Nagumo
python -u plot_submit1.py
python -u plot_submit2.py
python -u plot_image.py
cd ..

cd KolmogorovFlow
python -u plot.py
python -u plot_image.py
python -u plot_fno.py
python -u tradeoff.py
python -u tradeoffnew.py
cd ..

cd NS
python -u plot.py
python -u plot_image.py
cd ..


cd Basin
python -u plot.py
cd ..

cd Battery
python -u plot.py
python -u eval_cross_battery.py --model ani2 --checkpoint 2th/best_ani2_model.pth --data ./dataset/processed_battery_data_rollout.pt --predict_mode prior --use_meta_q0
python -u eval_cross_battery.py --model baseline --checkpoint baseline/best_baseline_model.pth --data ./dataset/processed_battery_data_rollout.pt --use_meta_q0
python -u eval_cross_battery.py --model ani2 --checkpoint 2th/best_ani2_model.pth --data ./dataset/processed_battery_data_rollout.pt --use_meta_q0
python -u eval_cross_battery.py --model ani4 --checkpoint 4th/best_ani4_model.pth --data ./dataset/processed_battery_data_rollout.pt --use_meta_q0
python -u eval_multicycle_standalone.py --model ani2 --checkpoint 2th/best_ani2_model.pth --data ./dataset/processed_battery_data_rollout.pt --out_csv ani2 --use_meta_q0
python -u eval_multicycle_standalone.py --model ani4 --checkpoint 4th/best_ani4_model.pth --data ./dataset/processed_battery_data_rollout.pt --out_csv ani4 --use_meta_q0
python -u eval_multicycle_standalone.py --model baseline --checkpoint baseline/best_baseline_model.pth --data ./dataset/processed_battery_data_rollout.pt --out_csv "baseline" --use_meta_q0
python -u plot_per_cycle_mse_compare.py
cd ..

cd HPC
python plot.py --csv loss_ani.csv --out ref_ani2.pdf --model "ANI-2"
python plot.py --csv loss_lie.csv --out ref_lie.pdf --model Lie 
cd ..
