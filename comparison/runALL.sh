echo "Running deepka..."
python comparison/deepka/1.collect.py
echo "Running pkai..."
python comparison/pkai/1.collect.py
sh comparison/pkai/2.run_pkai.sh
echo "Running pkai+..."
python comparison/pkai+/1.collect.py
sh comparison/pkai+/2.run_pkai+.sh
echo "Running propka..."
python comparison/propka/1.collect.py
sh comparison/propka/2.run_propka.sh
echo "Running pypka..."
python comparison/pypka/1.collect.py
./comparison/pypka/2.run_pypka.sh