import subprocess
import sys
from pathlib import Path

root = Path(__file__).parent
scripts = [
    root / "task1" / "task1_data_acquisition.py",
    root / "task2" / "task2_basic_statistics.py",
    root / "task3" / "task3_data_visualization.py",
]

for s in scripts:
    print(f"Running {s.name}...")
    r = subprocess.run([sys.executable, str(s)], cwd=str(root))
    if r.returncode != 0:
        sys.exit(1)

print("Done. Results in task1/, task2/, task3/")
