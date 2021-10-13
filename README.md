# Relighting Humans in the Wild: Monocular Full-Body Human Relighting with Domain Adaptation

This code is an implementation of the following paper:

Daichi Tajima, Yoshihiro Kanamori, Yuki Endo: "Relighting Humans in the Wild: Monocular Full-Body Human Relighting with Domain Adaptation," Computer Graphics Forum (Proc. of Pacific Graphics 2021), 2021. [[Project]](http://cgg.cs.tsukuba.ac.jp/~tajima/pub/relighting_in_the_wild/)[[PDF]]()

## Prerequisites
1. Python3
2. PyTorch(>=1.5)

## Demo
### Applying to images
To relight images under `PATH_TO_YOUR_INPUT_FOLDER`, run the following code:
```
sh ./scripts/demo_image.sh PATH_TO_YOUR_INPUT_FOLDER
```
The relighting results will be saved in `./demo/relighting_image/2nd` .

### Applying to videos
To relight video frames under `PATH_TO_YOUR_INPUT_FOLDER`, run the following code:
```
sh ./scripts/demo_video.sh PATH_TO_YOUR_INPUT_FOLDER
```
The flicker-tolerant relighting results will be saved in `./demo/relighting_video/flicker_reduction` .

## Citation
Please cite our paper if you find the code useful:
```
@article{tajimaPG21,
  author    = {Daichi Tajima,
               Yoshihiro Kanamori
               Yuki Endo},
  title     = {Relighting Humans in the Wild: Monocular Full-Body Human Relighting with Domain Adaptation},
  journal   = {Comput. Graph. Forum},
  volume    = {},
  number    = {},
  pages     = {--},
  year      = {2021},
}
```