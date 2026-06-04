import math
import numpy as np
from numba import cuda
import time
import signal
import os

# =========================================================
# MODE:
# 0 = don't print, don't beep
# 1 = only beep
# 2 = only print
# 3 = print and beep
# =========================================================
MODE = 0 

# Starting number
START = 20

# How many numbers processed per GPU batch
BATCH_SIZE = 500_000_000

THREADS_PER_BLOCK = 256

latest_prime = None
current_checked = 0
prime_count = 0

@cuda.jit
def mark_primes(start, results):
    i = cuda.grid(1)

    if i >= results.size:
        return

    n = start + i

    if n < 2:
        results[i] = 0
        return

    if n == 2 or n == 3:
        results[i] = 1
        return

    if n < 2 or n % 2 == 0 or n % 3 == 0:
        results[i] = 0
        return

    limit = int(math.sqrt(n))

    d = 5
    while d <= limit:
        if n % d == 0:
            results[i] = 0
            return
        if d + 2 <= limit and n % (d + 2) == 0:
            results[i] = 0
            return
        d += 6

    results[i] = 1


def beep():
    # Terminal beep
    print("\a", end="", flush=True)

def handle_status(signum, frame):
    global current_checked
    elapsed = time.time() - start_time

    if current_checked > 0:
        density = prime_count / current_checked
        rate = current_checked / elapsed
    else:
        density = 0
        rate = 0

    print(
            f"\n[STATUS] {elapsed:.2f} checked through {current_checked:,}, "
        f"total primes = {prime_count} ",
        f"density = {density:.4f} ",
        f"Rate={rate:,.0f}/sec  "
        f"latest prime = {latest_prime}\n",
        flush=True
    )

def process_batch(start_number):
    global latest_prime
    global prime_count
    global current_checked

    results = np.zeros(BATCH_SIZE, dtype=np.int32)
    d_results = cuda.to_device(results)

    blocks = (BATCH_SIZE + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
    mark_primes[blocks, THREADS_PER_BLOCK](start_number, d_results)

    results = d_results.copy_to_host()

    current_checked = start_number + BATCH_SIZE - 1

    prime_indices = np.where(results == 1)[0]

    if prime_indices.size > 0:
        latest_prime = start_number + int(prime_indices[-1])

    for i, is_prime in enumerate(results):
        if is_prime:
            n = start_number + i
            prime_count += 1

            if MODE in (2, 3):
                print(n)

            if MODE in (1, 3):
                beep()


def main():
    signal.signal(signal.SIGUSR1, handle_status)
    current = START
    global start_time;
    start_time = time.time();

    print(f"Start Time: {start_time}") 
    print(f"Process ID: {os.getpid()}") 
    print(f"BATCH_SIZE: {BATCH_SIZE}")
    print("GPU Prime Enumerator Started")
    print("Press Ctrl+C to stop.\n")
    print("Send status signal with:")
    print(f"kill -USR1 {os.getpid()}")

    try:
        while True:
            batch_start = time.time()
            process_batch(current)
            batch_elapsed = time.time() - batch_start
            batch_rate = BATCH_SIZE / batch_elapsed
            current += BATCH_SIZE
            elapsed = time.time() - start_time
            print(f"{elapsed:.2f} Processed through {current + BATCH_SIZE:,} batch rate = {batch_rate:,.0f}")

    except KeyboardInterrupt:
        print("\nStopped by user.")


if __name__ == "__main__":
    main()
