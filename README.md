# VanishCam Goku mode, macOS

A macOS port of the **Goku** (Instant Transmission) mode described in
[nicovic17/vanish_webcam](https://github.com/nicovic17/vanish_webcam). That
repo's README describes Goku mode, but its code isn't published, so this is a
reimplementation of the described behaviour, reusing the app's background
capture and person-segmentation approach.

## Run

```bash
./run_goku.sh                    # creates .venv, installs deps, downloads models
./run_goku.sh --camera 1         # another webcam
./run_goku.sh --no-virtual-camera   # preview window only
./run_goku.sh --gesture-debug    # print recognition details
```

1. Step out of frame for 4 seconds while the background is captured.
2. Step back in. Hold your **index and middle fingers straight and together
   against your forehead** (ring and little finger folded) for **1 second**.
   Vertical comic-style speed lines run across your silhouette, you squeeze
   into a bright line and vanish, and the lines linger briefly.
3. Lower your hand, then repeat the gesture for 1 second to come back (the
   reverse effect). Breaking the pose restarts the countdown.

Keys in the preview window: `SPACE` new background, `F` fullscreen, `Q`/`ESC` quit.
As in the original Goku mode, there is no manual `V`/`R` trigger.

## OBS

The output goes straight into **OBS Virtual Camera** (OBS Studio 30+ and its
camera system extension, which you activate by starting the virtual camera
once in OBS). **Stop** the virtual camera in OBS before running this app, then
pick "OBS Virtual Camera" in Zoom, Meet, FaceTime, and so on.

To use the effect inside an OBS scene instead, run with `--no-virtual-camera`
and add a **macOS Screen Capture → Window** source for "Webcam - Anteprima".

The first time you run it, macOS asks for camera access for your terminal app
(System Settings → Privacy & Security → Camera).

## Sound

Drop `assets/teleport.MP3` in to use your own sound. Otherwise a whoosh is
synthesized to `assets/teleport_synth.wav` and played with `afplay`.

## Tests

```bash
.venv/bin/python -m unittest test_goku
.venv/bin/python goku_cam.py --self-test
```
