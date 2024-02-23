# Dissertation-Tetris-AI

### Watching AI play:

1. Create and activate virtual env
```bash
    python3 -m venv env
    source env/bin/activate
```

2. Install pip packages in requirements.txt
```bash
    pip install -r requirements. txt
```

3. Run test_model.py 
```bash
    python3 test_model.py
```

To see additional parameters use ->
```bash
    python3 test_model.py --help
```

### Training AI model

1. Run train_model.py
```bash
    python3 train_model.py
```

2. View agent learning progress with bash command

First install tensorboard with ->
```bash
    pip install tensorboard==2.15.1
```

Then view a variety of graphs with ->

```bash
tensorboard --logdir=runs
```
or 
```bash
python tensorboard.main --logdir=runs
```
