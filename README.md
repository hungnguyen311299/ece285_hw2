# ECE 285 - Homework 2

Requires Python 3.10+.

A DC-GAN trained on 64x64 Impressionist landscape paintings from WikiArt.

## Data

From the
[`steubk/wikiart`](https://www.kaggle.com/datasets/steubk/wikiart) dataset,
copy the 4,365 Impressionism landscapes into `data/landscapes/`. These are the
rows of `wclasses.csv` whose `file` starts with `Impressionism/` and whose
`genre` is `133` (landscape). The images go directly in `data/landscapes/`,
without subfolders:

```
ece285_hw2/
├── data/
│   └── landscapes/
│       ├── <artist>_<title>.jpg
│       └── ...              (4,365 .jpg files)
├── main.py
├── metrics.py
├── model.py
└── requirements.txt
```

A fixed random 80% (3,493 images) is used for training.

## Run

```bash
python -m pip install -r requirements.txt
python main.py
```
