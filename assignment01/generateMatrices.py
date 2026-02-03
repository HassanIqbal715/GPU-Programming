import random
import sys
from pathlib import Path

# Default values in case of no arguments
n = 10
rows = 100
cols = 100

script_dir = Path(__file__).parent

# Takes max 3 arguments n rows cols.
if len(sys.argv) > 1:
    if len(sys.argv) == 2:
        n = int(sys.argv[1])
    elif len(sys.argv) == 3:
        n = int(sys.argv[1])
        rows = int(sys.argv[2])
    elif len(sys.argv) >= 4:
        n = int(sys.argv[1])
        rows = int(sys.argv[2])
        cols = int(sys.argv[3])

# Generates data at this specified path.
file_path = script_dir / "data" / "input.txt"
file_path.parent.mkdir(parents=True, exist_ok=True)

file = open(file_path, 'w')
file.write(str(n) + '\n')
file.write(str(rows) + " " + str(cols) + '\n')

for i in range(0, n):
    for j in range(0, rows):
        row_data = [str(random.randint(0, 100)) for _ in range(cols)]
        file.write(" ".join(row_data) + " ")
    file.write("\n")