# Dissertation-Tetris-AI

## Demo

![Tetris Demo](./TetrisDemo.gif)

## Setup

1. Create virtual env with python (tested working with version 3.10.5 but should work other recent python versions)
```
    py -m venv env
```

2. Activate virtual env
Command prompt
```
    venv/Scripts/activate
```

Posix 
```
source env/scripts/activate
```

3. Install pip packages in requirements.txt
```
    pip install -r requirements.txt
```

## Watching model play

Run test_model.py 
```
    py test_model.py
```
or alternatively, specify the size of the window and play speed:
```
    py test_model.py --size --speed
        e.g. 
            py test_model.py 13 10
```

## Training a new model

Run train_model.py
```
    py train_model.py
```

Optionally view agent learning progress with bash command:

Install tensorboard
```
    pip install tensorboard
```

Then view reward and duration values as graphs

```
    tensorboard --logdir=runs
```
