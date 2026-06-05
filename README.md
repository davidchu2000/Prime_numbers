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

For each batch, save progress status to save point file "prime_checkpoint.json"
if prime_checkpoint.json exists, it will start from last saved point
Sending Ctrl-C will also trigger to save the progress to save point file




Output log and status example:

Resuming from 22,500,000,019
Checkpoint saved: 23,000,000,019  1,008,309,536 primes
252.75 Processed through 23,500,000,020 session rate = 1,978,245 batch rate = 1,978,245 density = 0.0438 total primes = 1008309536 latest prime = 22999999987

[STATUS] 520.99 checked through 23,000,000,019, total primes = 1008309536  density = 0.0438  batch rate=1,978,245/sec  latest prime = 22999999987

Checkpoint saved: 23,500,000,019  1,029,256,462 primes
546.62 Processed through 24,000,000,020 session rate = 1,829,424 batch rate = 1,701,428 density = 0.0438 total primes = 1029256462 latest prime = 23500000001

[STATUS] 852.56 checked through 23,500,000,019, total primes = 1029256462  density = 0.0438  batch rate=1,701,428/sec  latest prime = 23500000001
