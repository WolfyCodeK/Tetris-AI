# Dissertation-Tetris-AI

## Watching AI play:

1. Create and activate virtual env
```bash
    python3 -m venv env
    source env/bin/activate
```

2. Install pip packages in requirements.txt
```
    pip install -r requirements. txt
```

3. Run test_model.py 
```
    python3 test_model.py
```

To see additional parameters use ->
```
    python3 test_model.py --help
```

## Training AI model

1. Run train_model.py
```
    python3 train_model.py
```

2. View agent learning progress with bash command

First install tensorboard with:
```
    pip install tensorboard==2.15.1
```

Then view a variety of graphs with:

```
tensorboard --logdir=runs
```
or 
```
python tensorboard.main --logdir=runs
```
