# Dissertation-Tetris-AI

## Demo

![Tetris Demo](./TetrisDemo.gif)

## Setup

1. Create and activate virtual env with python 3.10.5 (others versions may work but are untested)
```bash
    py -m venv env
    source env/scripts/activate
```

2. Install pip packages in requirements.txt
```bash
    pip install -r requirements.txt
```

## Watching model play

Run test_model.py 
```bash
    py test_model.py
```
or alternatively, specify the size of the window and play speed:
```bash
    py test_model.py --size --speed
        e.g. 
            py test_model.py 13 10
```

## Training a new model

Run train_model.py
```bash
    py train_model.py
```

Optionally view agent learning progress with bash command:

Install tensorboard
```bash
    pip install tensorboard
```

Then view reward and duration values as graphs

```bash
tensorboard --logdir=runs
```
or 
```bash
py tensorboard.main --logdir=runs
```
