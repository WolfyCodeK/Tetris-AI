# Dissertation-Tetris-AI

## Demo

![Tetris Demo](./TetrisDemo.gif)

## Setup

1. Create and activate virtual env
```bash
    python3 -m venv env
    source env/bin/activate
```

2. Install pip packages in requirements.txt
```bash
    pip install -r requirements. txt
```

## Watching model play

Run test_model.py 
```bash
    python3 test_model.py
```

## Training a new model

Run train_model.py
```bash
    python3 train_model.py
```

Optionally view agent learning progress with bash command:

Install tensorboard
```bash
    pip install tensorboard==2.15.1
```

Then view reward and duration values as graphs

```bash
tensorboard --logdir=runs
```
or 
```bash
python tensorboard.main --logdir=runs
```
