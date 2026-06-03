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
BATCH_SIZE = 10_000_000

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

    if n == 2:
        results[i] = 1
        return

    if n % 2 == 0:
        results[i] = 0
        return

    limit = int(math.sqrt(n))

    d = 3
    while d <= limit:
        if n % d == 0:
            results[i] = 0
            return
        d += 2

    results[i] = 1


def beep():
    # Terminal beep
    print("\a", end="", flush=True)

def handle_status(signum, frame):
    print(
        f"\n[STATUS] checked through {current_checked:,}, "
        f"total primes = {prime_count} ",
        f"density = {prime_count/latest_prime:.4f} ",
        f"latest prime = {latest_prime}\n",
        flush=True
    )

def process_batch(start_number):
    global latest_prime
    global prime_count

    results = np.zeros(BATCH_SIZE, dtype=np.int32)
    d_results = cuda.to_device(results)

    blocks = (BATCH_SIZE + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
    mark_primes[blocks, THREADS_PER_BLOCK](start_number, d_results)

    results = d_results.copy_to_host()

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
    global current_checked
    signal.signal(signal.SIGUSR1, handle_status)
    current = START

    print(f"Process ID: {os.getpid()}") 
    print("GPU Prime Enumerator Started")
    print("Press Ctrl+C to stop.\n")
    print("Send status signal with:")
    print(f"kill -USR1 {os.getpid()}")

    try:
        while True:
            process_batch(current)
            current_checked = current + BATCH_SIZE - 1
            print(f"Processed through {current + BATCH_SIZE:,}")

            current += BATCH_SIZE

    except KeyboardInterrupt:
        print("\nStopped by user.")


if __name__ == "__main__":
    main()
