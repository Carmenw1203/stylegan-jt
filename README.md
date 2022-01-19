# stylegan-jt
Stylegan Implemented by jittor in Computer Graphics（Fall 2021）, Tsinghua Univ.

## Train the model
I trained the stylegan model on a standard symbol dataset and FFHQ dataset.
You can get these two dataset using the following instructions.
```
wget -O data/color_symbol_7k.zip #standard symbol dataset
wget -O data/FFHQ_data.zip #standard symbol dataset
```
Then you can unzip the dataset and preprocess them using the following instructions to get data of different resolutions.
Don't forget to make the necessary modifications in [sh/preprocess.sh](sh/preprocess.sh) and [sh/preprocess_face.sh](sh/preprocess_face.sh) to adjust to your data path.
```
bash ./sh/preprocess.sh #standard symbol dataset
bash ./sh/preprocess_face.sh #standard symbol dataset
```

## Test the model

## Demo


## Reference

+ [pytorch Implementation: style-based-gan-pytorch](https://github.com/rosinality/style-based-gan-pytorch)
+ [StyleGAN: A Style-Based Generator Architecture for Generative Adversarial Networks](https://arxiv.org/abs/1812.04948)
+ [StyleGAN-jittor: another jittor implementation of stylegan from xUhEngwAng](https://github.com/xUhEngwAng/StyleGAN-jittor)
+ [Jittor: a novel deep learning framework with meta-operators and unified graph execution](https://cg.cs.tsinghua.edu.cn/jittor/papers/)
+ [Jittor Document](https://cg.cs.tsinghua.edu.cn/jittor/)
+ [StyleGAN - Official TensorFlow Implementation](https://github.com/NVlabs/stylegan)

## License
The files written by contributors in this project use [MIT LICENSE](LICENSE) open source without special instructions. This means that you may use, copy, modify, publish, and use the Project for commercial purposes at will, provided that this Open Source License must be included in all copies based on the Project and its derivatives.

The copyright for the remainder belongs to the respective authors.

