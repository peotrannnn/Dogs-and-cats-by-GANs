# sobelv5 Evolution Web — Local Test

This version uses the **Generator checkpoints** produced by
`04_model_training_5th_attempt.ipynb`.

## Copy into the project

```text
dog-gan-project/
├── models/
│   └── sobelv5/
│       ├── stage1/
│       │   └── snapshots/
│       ├── stage2/
│       │   └── snapshots/
│       └── stage3/
│           ├── snapshots/
│           └── best/
│               └── generator_best.weights.h5
├── src/
│   └── web_server.py
├── web/
│   ├── index.html
│   ├── style.css
│   └── script.js
└── requirements-web.txt
```

## Run

From `dog-gan-project/`:

```bash
pip install -r requirements-web.txt
python src/web_server.py
```

Open:

```text
http://127.0.0.1:5000
```

You can inspect the exact checkpoint sequence at:

```text
http://127.0.0.1:5000/api/health
```

## Evolution mechanism

One click creates exactly one latent vector:

```text
z ~ N(0,1)
```

and one class label:

```text
dog = 0
cat = 1
```

That **same z and same class** are run through historical Generator weights.

The v5 training notebook stores permanent Generator weights at these cadences:

```text
stage1: every 25 epochs
stage2: every 25 epochs
stage3: every 10 epochs
```

For stage2/stage3, the web server prefers EMA snapshots because those stages
use the EMA Generator for sampling/evaluation.

A representative subset is used for the animation:

```text
Stage 1: ~25, 50, 75, 100, 125, 150
Stage 2: ~25, 75, 125, 200
Stage 3: ~10, 30, 60, 100
Final:   stage3/best/generator_best.weights.h5
```

The server automatically selects the nearest checkpoint that actually exists,
so Stage 3 still works if early stopping ended it before epoch 100.

Each generated frame is streamed to the browser as soon as its checkpoint has
finished inference. The browser crossfades frames instead of waiting for the
whole sequence.

## Why there is no epoch 1 / 5 / 10 Generator for Stage 1

In v5, every 5 epochs the notebook saves:

- the sample grid image;
- the rolling Generator checkpoint.

The rolling checkpoint is overwritten, so it does not preserve the historical
epoch-5 model.

The permanent historical Generator weights are the `snapshots/` files. Stage 1
and Stage 2 save those every 25 epochs; Stage 3 saves them every 10 epochs.

To support arbitrary future web seeds, actual historical weights are required.
Saved sample-grid PNG files cannot generate a new arbitrary latent `z`.

## Later GitHub Pages

This local version still requires Flask + Python TensorFlow.

The frontend is plain HTML/CSS/JavaScript. Later, after converting selected
Generator checkpoints to TensorFlow.js, the same evolution concept can run
entirely in the browser on GitHub Pages.
