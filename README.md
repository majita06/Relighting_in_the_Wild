# Relighting Humans in the Wild: Monocular Full-Body Human Relighting with Domain Adaptation

This code is an implementation of the following paper:

Daichi Tajima, Yoshihiro Kanamori, Yuki Endo: "Relighting Humans in the Wild: Monocular Full-Body Human Relighting with Domain Adaptation," Computer Graphics Forum (Proc. of Pacific Graphics 2021), 2021. [[Project]](http://cgg.cs.tsukuba.ac.jp/~tajima/pub/relighting_in_the_wild/)[[PDF]]()

## Prerequisites
1. Python3
2. PyTorch(>=1.5)

## Demo

Relighting images
```
sh ./scripts/demo_image.sh PATH_TO_YOUR_INPUT_FOLDER
```

Relighting videos
```
sh ./scripts/demo_video.sh PATH_TO_YOUR_INPUT_FOLDER
```

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