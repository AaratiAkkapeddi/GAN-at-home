# Lightweight GAN Kit

Train a small, unconditional GAN from scratch on your own photos (roughly
600–1000 of them), entirely on your laptop, then run the finished
generator live in a web page.

Everything is driven from the command line. You will not need to write
any code to train a model.


## What's in the box

```
src/
  prepare_data.py   clean up your raw photos before training
  train.py          the main training loop
  generate.py       sample images from a checkpoint, no training needed
  generate_walk.py  render a smooth latent-space walk as an .mp4 video
  export_onnx.py    checkpoint -> .onnx
  model.py          generator + discriminator definitions
  dataset.py        image loading
  diffaug.py        data augmentation used during training
  utils.py          small shared helpers
web_demo/
  index.html   a live in-browser demo where you can upload your .onnx model
data/raw/           put your original photos here
requirements.txt          training environment
requirements-export.txt   export-to-web environment (separate, optional)
SETUP_MAC.md          Apple Silicon setup instructions
```

## Quick start

### 1. Set up your environment

See `SETUP_MAC.md`. TLDR run: `python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`.

### 2. Add your photos

Drop 600–1000+ images into `data/raw/`. They don't need to be the same
size or aspect ratio. 

Clean them up into a standard format:

```bash
python src/prepare_data.py --input data/raw --output data/processed --size 256
```

This converts everything to RGB JPEG, and resizes 


### 3. Train model

```bash
python src/train.py --data data/processed --out runs/my_run --img_size 256
```

Leave this running. Check `runs/my_run/samples/` every so often — a new
grid image is written every `--sample_every` iterations (default 200) so
you can watch progress without interrupting training. Stop any time with
`Ctrl+C`; it saves a checkpoint before exiting, and you can resume with:

```bash
python src/train.py --data data/processed --out runs/my_run --img_size 256 \
    --resume runs/my_run/checkpoints/interrupted.pt
```

**How long does this take?** There's no fixed answer -- it depends on the
dataset and the laptop — but for 256px on an M4 Mac, plan on this being a
multi-hour (likely overnight) job. Run it in a
terminal you can leave open (see `SETUP_MAC.md` for keeping the laptop
awake).

### 4. Generate images from a checkpoint

```bash
python src/generate.py --checkpoint runs/my_run/checkpoints/latest.pt --num_images 16
```
Writes a grid image plus individual PNGs to `samples_out/`.

### 5. Bonus: render a latent-space walk video

smoothly morph between random generated images by 'walking' through latent space and rendering every step as a video frame.

```bash
python src/generate_walk.py --checkpoint runs/my_run/checkpoints/latest.pt \
    --out walk.mp4 --num_anchors 8 --steps_per_segment 30 --fps 24
```

That's 8 random points with 30 interpolated frames between each
(240 frames, 10 seconds at 24fps), looping back to the start so it plays
seamlessly on repeat. `--truncation` (default 0.8) controls how "safe"
vs. varied the walk looks. Lower it if the morph looks erratic, raise it
toward 1.0 for more variety. It needs one extra dependency already listed
in `requirements.txt`: `imageio` + `imageio-ffmpeg` (the latter bundles
its own ffmpeg binary, so nothing extra to install system-wide).

### 6. Export for the web

```bash
pip install --upgrade pip
pip install -r requirements-export.txt   # separate, heavier environment
python src/export_onnx.py --checkpoint runs/my_run/checkpoints/latest.pt --out exported/model.onnx
```

Then just open `web_demo/index.html` in a browser (double-click it, or
`python3 -m http.server` from that folder and visit `localhost:8000`

## Choosing settings for your laptop

Unified memory is the main constraint on Mac. As a starting point:

| RAM   | Suggested `--img_size` | Suggested `--batch_size` |
|-------|------------------------|---------------------------|
| 8 GB  | 128                    | 4                          |
| 16 GB | 256                    | 8 (default)                |
| 24 GB+| 256                    | 16                         |

If training crashes with a memory error, lower `--batch_size` first,
then `--img_size`.

## Reading the training log

Each logged line looks like:

```
[  2000/40000] D: 0.62 (recon 0.31)  G: 1.10  1840s elapsed
```

- **D** and **G** are the discriminator/generator losses. GAN losses are
  notoriously unintuitive to read directly. Don't expect them to
  smoothly decrease to zero. What matters is that neither number runs
  away to a huge value and neither collapses to exactly 0, and that the
  sample images keep visibly improving.
- **recon** is the discriminator's self-supervised reconstruction loss.
  This one *should* trend down over training. If it stays high and flat,
  something (usually too small/low-variety a dataset) is not clicking.

## Troubleshooting / what can go wrong

- **Samples stay blurry mush past a few thousand iterations:** try more
  images and double-check `prepare_data.py` didn't silently skip most of your files.
- **Samples look identical to each other (mode collapse):** the model
  found one shortcut output that fools the discriminator. Try lowering
  `--lr`, or check your dataset isn't dominated by near-duplicate images 
  (aka add more variety to your images).
- **Samples start looking great, then degrade later in training:** this
  is normal GAN instability — go back to an earlier checkpoint in
  `runs/.../checkpoints/`, they're kept periodically (`--keep_last` controls how many).
- **`RuntimeError` mentioning MPS / an unsupported operator:** the script
  already sets `PYTORCH_ENABLE_MPS_FALLBACK=1`, which should route
  unsupported ops to CPU automatically. If it still crashes, try `--device
  cpu` to confirm whether it's MPS-specific, then report the exact error to me over email. 
  You may want to try the colab version instead because maybe your laptop doesn't support running this locally.
- **DataLoader worker errors on Mac:** try adding `--workers 0` to the train command.
- **The web export step fails:** this conversion chain (ONNX) is the most fragile part of this kit because those
  tools update frequently. Email me if you are running into issues with this part. Worse come to worse, we can convert your .pth file on my laptop.
- **Bigger/smaller images:** `--img_size` also accepts `64` and `128` if
  you want a faster/lower-memory option.
