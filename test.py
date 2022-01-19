import numpy as np
import math
import jittor as jt
from model.model import StyledGenerator
jt.flags.use_cuda = True
import argparse
import pathlib
import os

@jt.no_grad()
def style_mixing(generator, step, mean_style, n_source, n_target):
    source_code = jt.randn(n_source, 512)
    target_code = jt.randn(n_target, 512)
    
    shape = 4 * 2 ** step
    alpha = 1

    images = [jt.ones((1, 3, shape, shape)) * -1]

    source_image = generator(
        source_code, step=step, alpha=alpha, mean_style=mean_style, style_weight=0.7
    )
    target_image = generator(
        target_code, step=step, alpha=alpha, mean_style=mean_style, style_weight=0.7
    )

    images.append(source_image)
    # print(source_code.shape)
    for i in range(n_target):
        image = generator(
            [target_code[i].unsqueeze(0).repeat(n_source, 1), source_code],
            step=step,
            alpha=alpha,
            mean_style=mean_style,
            style_weight=0.7,
            mixing_range=(0, 1),
        )
        images.append(target_image[i].unsqueeze(0))
        images.append(image)

    images = jt.concat(images, 0)
    # print(images.shape)
    return images

def inference(args):
    #init
    generator = StyledGenerator(512)
    ckpt = jt.load(args.ckpt)
    generator.load_state_dict(ckpt)
    generator.eval()

    mean_style = None
    for i in range(10):
        with jt.no_grad():
            style = generator.mean_style(jt.randn(1024,512))
        if mean_style is None:
            mean_style = style
        else:
            mean_style += style
    mean_style /= 10
    step = int(math.log(args.resolution, 2)) - 2

    first_code = jt.randn(50, 512)
    last_code = jt.randn(50, 512)
    inter_times = 2000
    with jt.no_grad():
        first_img = generator(
            first_code,
            step=step,
            alpha=1,
            mean_style=mean_style,
            style_weight=0.7,
        )
    with jt.no_grad():
        last_img = generator(
            first_code,
            step=step,
            alpha=1,
            mean_style=mean_style,
            style_weight=0.7,
        )
    delta_code = (last_code - first_code)/inter_times
    pathlib.Path(args.output_dir).mkdir(parents=True,exist_ok=True)
    pathlib.Path(args.stylemixing_dir).mkdir(parents=True,exist_ok=True)
    for i in range(inter_times):
        image = generator(
            first_code + delta_code*i,
            step=step,
            alpha=1,
            mean_style=mean_style,
            style_weight=0.7,
            # mixing_range=(0, 1),
        )
        jt.save_image(
            image, os.path.join(args.output_dir,f'sample_{i}.png'), nrow=10, normalize=True, range=(-1, 1)
        )
    
    with jt.no_grad():
        img = generator(
            jt.randn(25, 512),
            step=step,
            alpha=1,
            mean_style=mean_style,
            style_weight=0.7,
        )
    jt.save_image(img, os.path.join(args.stylemixing_dir,f'sample.jpg'), nrow=5, normalize=True, range=(-1, 1))
    print(img[0,:,0,0])
    for j in range(20):
        img = style_mixing(generator, step, mean_style, 5, 10)
        jt.save_image(
            img, os.path.join(args.stylemixing_dir,f'sample_mixing_{j}.jpg'), nrow=5 + 1, normalize=True, range=(-1, 1)
        )
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', default='./checkpoints/symbol_80w_ckpt/800000.model',type=str,help='checkpoint path')
    parser.add_argument('--resolution',default=128,type=int)
    parser.add_argument('--output_dir',default='./output/interpolation_80_80w',type=str)
    parser.add_argument('--stylemixing_dir',default='./output/style_mixing_80_80w',type=str)
    args = parser.parse_args()
    
    inference(args)
    
