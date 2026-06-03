# Prime_number generator
# This program generate prime numbers


# This program utilize Nvidia GPU so the conda environment need to be installed and set up
# set up conda
conda info --env
conda activate numba-cuda

# To run
> python prime_run.py


# =========================================================
# MODE:
# 0 = don't print, don't beep
# 1 = only beep
# 2 = only print
# 3 = print and beep
# =========================================================

in mode 0, sending signal USR1 to process to dump the latest number
> kill -USR1 PID

after sending USR1 signal, it will print out the latest progress:

[STATUS] checked through 24,110,000,019, total primes = 1054788835  density = 0.0437  latest prime = 24110000017
