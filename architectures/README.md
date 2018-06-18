Implementation and benchmark of well-known deep learning architectures in Keras

# Benchmark

|              |    Params   |  Load | Pred (CPU) | Pred (GPU) |
|:------------:|:-----------:|:-----:|:----------:|:----------:|
|    VGG-16    | 138,357,544 |  3.92 |    1.14    |    0.27    |
|    VGG-19    | 143,667,240 |  4.36 |    1.48    |    0.37    |
|   ResNet50   |  25,636,712 |  9.33 |    0.96    |    1.13    |
|   MobileNet  |  4,253,864  |  4.45 |    0.51    |    0.61    |
| Inception v3 |  23,851,784 | 13.60 |    1.30    |    2.28    |
|   Xception   |  22,910,480 |  8.50 |    2.16    |    1.60    |

* all times are presented in seconds