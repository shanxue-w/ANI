#!/usr/bin/env bash
set -ex

pip install -e .

cd Pendulum
python -u plot.py
cd ..

cd Lorenz-stenflo
python -u plot.py
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
cd ..

cd KolmogorovFlow
python -u plot.py
python -u plot_image.py
cd ..

cd NS
python -u plot.py
python -u plot_image.py
cd ..
