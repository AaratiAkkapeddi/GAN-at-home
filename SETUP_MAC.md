# Setup on M-series Macs (M4 first)

## 1. Install Python

You need Python 3.10 or 3.11. Check what you have:

```bash
python3 --version
```

If you don't have one of those, install via [python.org](https://www.python.org/downloads/macos/) or Homebrew:

```bash
brew install python@3.11
```

## 2. Create a virtual environment

From inside this repo folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
```


## 3. Install PyTorch with Metal (MPS) support

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

. Confirm the Metal backend is visible:

```bash
python3 -c "import torch; print(torch.backends.mps.is_available())"
```

This should print `True`. If it prints `False`, your torch install didn't pick up MPS support — try reinstalling in a fresh venv, and make sure you're not accidentally running under Rosetta (an Intel-emulated terminal). Check with:

```bash
python3 -c "import platform; print(platform.machine())"
```

This should print `arm64`, not `x86_64`.

## 4. Keep your Mac awake during training

Training will run for a while. Prevent your Mac from sleeping partway through with `caffeinate`, run in a separate terminal tab alongside training:

```bash
caffeinate -i
```

(`-i` prevents idle sleep; leave that tab open for the whole training run. Closing the lid will still sleep the machine though. Make sure to plug in and adjust Energy Saver settings if you're doing a long overnight run.)

## 5. Non-M4 / non-Apple-Silicon laptops

Use the colab version!
