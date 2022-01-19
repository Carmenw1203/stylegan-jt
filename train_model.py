import os
import argparse
import numpy as np
import jittor as jt
from model.model import StyledGenerator, Discriminator
from dataset.dataset import MultiResolutionDataset
import math
from tqdm import tqdm
import random
import pathlib

jt.flags.use_cuda = True
jt.flags.log_silent = True

my_bs_dict = {
    4: 128,
    8: 128, 
    16: 32, 
    32: 8, 
    64: 2, 
    128: 1
}

def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag

def train_model(args):
    # init model
    code_size = 512
    my_generator = StyledGenerator(code_dim=code_size)
    optimizer_G = jt.optim.Adam(my_generator.generator.parameters(), lr=args.lr, betas=(0.0, 0.99))
    optimizer_G.add_param_group({
        'params': my_generator.style.parameters(),
        'lr': args.lr * 0.01,
        'mult': 0.01,
        }
    )
    g_running = StyledGenerator(code_size)
    g_running.eval()
    my_discriminator = Discriminator(from_rgb_activate=True)
    optimizer_D = jt.optim.Adam(my_discriminator.parameters(), lr=args.lr, betas=(0.0, 0.99))

    print('resuming from checkpoint .......')
    # init loss
    disc_loss_val = 0
    gen_loss_val = 0
    grad_loss_val = 0

    alpha = 0
    step = 1
    used_sample = 0
    phase = 150_000
    final_progress = False
    pbar = tqdm(range(800_000))
    max_step = int(math.log2(args.max_reso)) - 2
    # accumulate
    dict_g_r = dict(g_running.named_parameters())
    dict_my_g = dict(my_generator.named_parameters())
    for k in dict_g_r.keys():
        dict_g_r[k].update(dict_my_g[k].detach())

    # training
    requires_grad(my_generator, False)
    requires_grad(my_discriminator, True)
    img_dataset = MultiResolutionDataset(args.data_src,args.start_reso)#start from resolution = 8
    data_loader = img_dataset.set_attrs(batch_size=my_bs_dict[args.start_reso], shuffle=True)
    train_loader = iter(data_loader)
    # print(my_bs_dict[args.start_reso])

    for i in pbar:
        alpha = min(1, 1 / phase * (used_sample + 1))
        if (step == 1) or final_progress:
            alpha = 1
        if used_sample > phase * 2:
            used_sample = 0
            step += 1

            if step > max_step:
                step = max_step
                final_progress = True
                ckpt_step = step + 1
            else:
                alpha = 0
                ckpt_step = step

            resolution = 4 * 2 ** step

            img_dataset = MultiResolutionDataset(args.data_src, resolution)
            data_loader = img_dataset.set_attrs(
                batch_size=my_bs_dict[resolution], 
                shuffle=True
            )
            train_loader = iter(data_loader)
            jt.save(
                {
                    'generator': my_generator.state_dict(),
                    'discriminator': my_discriminator.state_dict(),
                    'g_running': g_running.state_dict(),
                },
                os.path.join(args.ckpt,f'train_step-{ckpt_step}.model'),
            )
        try:
            real_img = next(train_loader)
        except (OSError, StopIteration):
            train_loader = iter(data_loader)
            real_img = next(train_loader)
        
        real_img.requires_grad = True
        b_size = real_img.size(0)

        real_scores = my_discriminator(real_img, step=step, alpha=alpha)
        real_predict = jt.nn.softplus(-real_scores).mean()

        grad_real = jt.grad(real_scores.sum(), real_img)
        grad_penalty = (
            grad_real.reshape(grad_real.size(0), -1).norm(2, dim=1) ** 2
        ).mean()
        grad_penalty = 10 / 2 * grad_penalty

        if i % 10 == 0:
            grad_loss_val = grad_penalty.item()

        if args.mixing and random.random() < 0.9:
            gen_in11, gen_in12, gen_in21, gen_in22 = jt.randn(4, b_size, code_size).chunk(4, 0)
            gen_in1 = [gen_in11.squeeze(0), gen_in12.squeeze(0)]
            gen_in2 = [gen_in21.squeeze(0), gen_in22.squeeze(0)]
        else:
            gen_in1, gen_in2 = jt.randn(2, b_size, code_size).chunk(2, 0)
            gen_in1 = gen_in1.squeeze(0)
            gen_in2 = gen_in2.squeeze(0)

        fake_image = my_generator(gen_in1, step=step, alpha=alpha)
        fake_predict = my_discriminator(fake_image, step=step, alpha=alpha)
        fake_predict = jt.nn.softplus(fake_predict).mean()

        if i % 10 == 0:
            disc_loss_val = (real_predict + fake_predict).item()

        loss_D = real_predict + grad_penalty + fake_predict
        optimizer_D.step(loss_D)

        # optimize generator

        requires_grad(my_generator, True)
        requires_grad(my_discriminator, False)

        fake_image = my_generator(gen_in2, step=step, alpha=alpha)
        predict = my_discriminator(fake_image, step=step, alpha=alpha)
        loss_G = jt.nn.softplus(-predict).mean()

        if i % 10 == 0:
            gen_loss_val = loss_G.item()

        optimizer_G.step(loss_G)

        dict_g_r = dict(g_running.named_parameters())
        dict_my_g = dict(my_generator.named_parameters())
        for k in dict_g_r.keys():
            dict_g_r[k].update(dict_g_r[k] * 0.999 + 0.001 * dict_my_g[k].detach())
        requires_grad(my_generator, False)
        requires_grad(my_discriminator, True)

        used_sample += real_img.shape[0]

        if (i + 1) % 100 == 0:
            images = []

            gen_i, gen_j = (10, 5)

            with jt.no_grad():
                for _ in range(gen_i):
                    images.append(
                        g_running(
                            jt.randn(gen_j, code_size), step=step, alpha=alpha
                        ).data
                    )
            print(images[0][0,:,0,0])
            pathlib.Path(args.output_dir).mkdir(parents=True,exist_ok=True)
            pathlib.Path(args.ckpt).mkdir(parents=True,exist_ok=True)
            jt.save_image(
                jt.concat(images, 0),
                os.path.join(args.output_dir,f'{str(i + 1).zfill(6)}.png'),
                nrow=gen_i,
                normalize=True,
                range=(-1, 1),
            )

        if (i + 1) % 10000 == 0:
            jt.save(g_running.state_dict(), os.path.join(args.ckpt,f'{str(i + 1).zfill(6)}.model'))

        state_msg = (
            f'Size: {4 * 2 ** step}; G: {gen_loss_val:.3f}; D: {disc_loss_val:.3f};'
            f' Grad: {grad_loss_val:.3f}; Alpha: {alpha:.5f}'
        )
        pbar.set_description(state_msg)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training Stylegan')
    parser.add_argument('--data_src', default='./data/symbol_data/color_symbol_7k_',type=str, help='path of src dataset')
    parser.add_argument('--ckpt', default='./checkpoints/symbol_80w_ckpt',type=str, help='save path of checkpoints')
    parser.add_argument('--output_dir', default='./output/symbol_80w',type=str, help='save result pics')
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--start_reso',default=8,type=int,help='start resolution')
    parser.add_argument('--max_reso',default=128,type=int,help='max resolution')
    parser.add_argument('--mixing',default=True,type=bool)
    args = parser.parse_args()
    train_model(args)

