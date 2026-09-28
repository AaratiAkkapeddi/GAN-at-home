Colab version for those that can't run this training locally: [https://colab.research.google.com/drive/1Oa_tpAJHz3oqGnbk0KL-LyGxFQX60bOi?usp=sharing](https://colab.research.google.com/drive/1Oa_tpAJHz3oqGnbk0KL-LyGxFQX60bOi?usp=sharing)


# Lightweight GAN Kit

Train a small, unconditional GAN from scratch on your own images (600 minimum of them), entirely on your laptop, then run the finished
generator live in a web page.


## Important files

```
src/
  prepare_data.py   clean up your raw photos before training
  train.py          the main training loop
  generate.py       generate images w trained model
  generate_walk.py  render a smooth latent-space walk as an .mp4 video
  export_onnx.py    convert your model checkpoints to .onnx files which can be used on the web
web_demo/
  index.html   a live in-browser demo where you can upload your .onnx model
data/raw/           put your dataset in here
```

## Quick start

### 1. Set up your environment

Depending on your python setup you may need to replace `python3` with just plain old `python` in the terminal commands throughout this document. You may also need to install python [https://www.python.org/downloads/](https://www.python.org/downloads/)!

In the terminal window -  
Run: `python3 -m venv .venv`
Run: `source .venv/bin/activate`
Run: `pip install -r requirements.txt`

### 2. Add your photos

Drop 600–1000+ images into `data/raw/`. 

Clean them up into a standard format by running:

```bash
python src/prepare_data.py --input data/raw --output data/processed --size 256
```

This converts everything to RGB JPEG, crops them square (by the center) and resizes them to 256x256

If you want to augment your dataset by flipping each image horizontally just add the `--flip-horizontal` flag like so..

```bash
python src/prepare_data.py --input data/raw --output data/processed --size 256 --flip_horizontal
```

### 3. Train model

```bash
python src/train.py --data data/processed --out runs/my_run --img_size 256
```

Leave this running. This will take many hours so I leave my laptop plugged into a power source. You could run it overnight or while you are doing other things not on your laptop. You can check `runs/my_run/samples/` every so often to see how its going. A new grid image is written every `--sample_every` iterations (default 200) so you can watch progress without interrupting training. 

*Also keep in mind how much space you have on your laptop.* Each checkpoint is about 280MB and so you may need to delete older checkpoints to keep your computer going or lower the frequency of checkpoint saving. By default it saves every 1000 checkpoints but you can increase this number using the --ckpt_every flag (i.e. `--ckpt_every 2000` to save every 2000 instead of 1000, saving you space but giving you less of a granular history of training)

Stop any time with
`Ctrl+C`; it saves a checkpoint before exiting, and you can resume with:

```bash
python src/train.py --data data/processed --out runs/my_run --img_size 256 \
    --resume runs/my_run/checkpoints/interrupted.pt
```

### 4. Generate images from a checkpoint

```bash
python src/generate.py --checkpoint runs/my_run/checkpoints/latest.pt --num_images 16
```
Writes a grid image plus individual PNGs to `samples_out/`.

### 5. Render a latent-space walk video

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

### 6. Export the trained model for use on the web
In the terminal run:

```bash
pip install --upgrade pip
pip install -r requirements-export.txt  
python src/export_onnx.py --checkpoint runs/my_run/checkpoints/latest.pt --out exported/model.onnx
```

Then just open `web_demo/index.html` in a browser (double-click it, or
`python3 -m http.server` from that folder and visit `localhost:8000`. If you have never run a website locally before please let me know. Happy to give you a quick demo :)

Alternatively you can just go to the online version at [https://handmadedatasets.com/lightweight_gan/](https://handmadedatasets.com/lightweight_gan/)

## Choosing settings for your laptop


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

- **D** and **G** are the discriminator/generator losses. The discriminator loss measures how well the discriminator distinguishes real data from fake data, while the generator loss measures how well the generator tricks the discriminator into believing its fake data is real. Generally speaking, you are looking for a general decrease over time in both. There might be some slight jumping around but what you *don't want* is a huge increase randomly or a collapse into 0. Personally, I find it really hard for me to assess using these numbers and so I instead look at the visual outputs from the samples. 

## Troubleshooting / what can go wrong

- **Samples stay blurry mush past a few thousand iterations:** Your images might be too varied for the model to grasp any patterns and so you can either increase the amount of data or figure out creative ways of "homogenizing" the dataset further (like removing and norming the background).
- **Samples look identical to each other (mode collapse):** the model
  found one shortcut output that fools the discriminator. Try lowering
  `--lr`, or check your dataset isn't dominated by near-duplicate images 
  (aka add more variety to your images).
- **Samples start looking great, then degrade later in training:** It could be that your model just trained really fast and you kind of *overtrained it*. You can always go back to an earlier checkpoint in  `runs/.../checkpoints/`, they're kept periodically (`--keep_last` controls how many).
- **`RuntimeError` mentioning MPS / an unsupported operator:** 
  You should use the colab version instead because maybe your laptop doesn't support running this locally.
- **DataLoader worker errors on Mac:** try adding `--workers 0` to the train command.
- **The web export step fails:**  Email me if you are running into issues with this part. Worse come to worse, we can convert your model file on my laptop.
- **Bigger/smaller images:** `--img_size` also accepts `64` and `128` if
  you want a faster/lower-memory option.
