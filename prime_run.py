import math
import numpy as np
from numba import cuda
import time
import signal
import os
import json
import os

# =========================================================
# MODE:
# 0 = don't print, don't beep
# 1 = only beep
# 2 = only print
# 3 = print and beep
# =========================================================
MODE = 2

# Starting number
START = 20
#START = np.int64(10**18)   # 10^18,  1 quintillion , cuda integer max 9.22 × 10^18
#START = np.int64(9223372036854775807-BATCH_SIZE*2) #max int64 minus 2 BATCH_SIZE:

# How many numbers processed per GPU batch
BATCH_SIZE = 100_000 
#BATCH_SIZE = 500_000_000
#BATCH_SIZE = 100 
#BATCH_SIZE = 1_000_000

THREADS_PER_BLOCK = 256

current_checked = 0
batch_rate = 0

# save check point
current_checked = 0
latest_prime = 0
total_primes = 0
CHECKPOINT_FILE = "prime_checkpoint.json"

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
    global batch_rate
    elapsed = time.time() - start_time

    if current_checked > 0:
        density = total_primes / (current_checked - START)
    else:
        density = 0
        batch_rate = 0

    print(
            f"\n[STATUS] {elapsed:.2f} checked through {current_checked:,}, "
        f"total primes = {total_primes} ",
        f"density = {density:.4f} ",
        f"batch rate={batch_rate:,.0f}/sec  "
        f"latest prime = {latest_prime}\n",
        flush=True
    )

def process_batch(start_number):
    global latest_prime
    global total_primes 
    global current_checked

    results = np.zeros(BATCH_SIZE, dtype=np.int32)
    d_results = cuda.to_device(results)

    blocks = (BATCH_SIZE + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
    mark_primes[blocks, THREADS_PER_BLOCK](start_number, d_results)

    results = d_results.copy_to_host()

    current_checked = int(start_number) + int(BATCH_SIZE) - 1

    prime_indices = np.where(results == 1)[0]

    prev_n = latest_prime

    if prime_indices.size > 0:
        latest_prime = int(start_number) + int(prime_indices[-1])

    for i, is_prime in enumerate(results):
        if is_prime:
            n = start_number + i
            total_primes += 1

            if MODE in (2, 3):
                if (n - prev_n) == 2:
                    print(f"{n} Twin\n")
                else:
                    print(n)

            if MODE in (1, 3):
                beep()

            prev_n = n


def save_checkpoint():
    data = {
        "current_checked": int(current_checked),
        "latest_prime": int(latest_prime),
        "total_primes": int(total_primes)
    }

    with open("checkpoint.tmp", "w") as f:
        json.dump(data, f)

    # make the output atomic
    os.replace("checkpoint.tmp", CHECKPOINT_FILE)
    print(
        f"Checkpoint saved: "
        f"{current_checked:,}  "
        f"{total_primes:,} primes",
        flush=True
    )

def load_checkpoint():
    global current_checked
    global latest_prime
    global total_primes

    if not os.path.exists(CHECKPOINT_FILE):
        return False

    with open(CHECKPOINT_FILE) as f:
        data = json.load(f)

    current_checked = data["current_checked"]
    latest_prime = data["latest_prime"]
    total_primes = data["total_primes"]

    return True

def main():
    signal.signal(signal.SIGUSR1, handle_status)
    current = START
    global start_time;
    global batch_rate;
    start_time = time.time();

    print(f"Start Time: {start_time}") 
    print(f"Process ID: {os.getpid()}") 
    print(f"BATCH_SIZE: {BATCH_SIZE}")
    print("GPU Prime Enumerator Started")
    print("Press Ctrl+C to stop.\n")
    print("Send status signal with:")
    print(f"kill -USR1 {os.getpid()}")

    # Load check point
    if load_checkpoint():
        current = current_checked + 1
        resume_from = current_checked

        print(
           f"Resuming from "
           f"{current_checked:,}"
        )
    else:
        current = START
        resume_from = current

    run_start_time = time.time()
    try:
        while True:
            batch_start = time.time()
            process_batch(current)
            batch_elapsed = time.time() - batch_start
            batch_rate = BATCH_SIZE / batch_elapsed
            current += BATCH_SIZE
            elapsed_since_resume = time.time() - run_start_time
            session_rate = (current_checked - resume_from) / elapsed_since_resume
            density = total_primes / (current_checked - START)
            print(f"{elapsed_since_resume:.2f} Processed through {current + BATCH_SIZE:,} session rate = {session_rate:,.2f} batch rate = {batch_rate:,.2f} density = {density:.4f} total primes = {total_primes} latest prime = {latest_prime}", flush=True)
            save_checkpoint()

    except KeyboardInterrupt:
        print("\nStopped by user.")
        save_checkpoint()
        print("Checkpoint written.")


if __name__ == "__main__":
    main()
