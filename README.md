# ECE 285 - Homework 1

Requires Python 3.10+.

```bash
cd ece285_hw1
python -m pip install -r requirements.txt
python main.py --epochs 100
```

The dataset downloads automatically. Results are saved in `results/`.

If you already have the dataset:

```bash
python main.py --data-path /path/to/chest_xray --epochs 100
