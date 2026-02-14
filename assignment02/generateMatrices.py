import random
import sys
from pathlib import Path

# Default values in case of no arguments
n = 2
rows = [100, 100]
cols = [100, 200]

script_dir = Path(__file__).parent

# Takes max 3 arguments n rows cols.
if len(sys.argv) > 1:
    if len(sys.argv) == 3:
        rows[0] = int(sys.argv[1])
        cols[0] = int(sys.argv[2])
    elif len(sys.argv) == 5:
        rows[0] = int(sys.argv[1])
        cols[0] = int(sys.argv[2])
        rows[1] = int(sys.argv[3])
        cols[1] = int(sys.argv[4])

# Generates data at this specified path.
file_path = script_dir / "data" / "input.txt"
file_path.parent.mkdir(parents=True, exist_ok=True)

file = open(file_path, 'w')
file.write(str(n) + '\n')

for i in range(0, n):
    file.write(str(rows[i]) + " " + str(cols[i]) + '\n')
    row_data = [str(random.randint(0, 100)) for _ in range(cols[i] * rows[i])]
    file.write(" ".join(row_data) + " ")
    file.write("\n")