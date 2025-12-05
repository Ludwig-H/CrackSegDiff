import argparse
import os
import sys
import random
sys.path.append(".")
import numpy as np
from tqdm import tqdm
import torch as th
from PIL import Image
from guided_diffusion import dist_util, logger
from guided_diffusion.custom_dataset_loader import CustomDataset
from guided_diffusion.utils import staple
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
import torchvision.transforms as transforms
seed=10
th.manual_seed(seed)
th.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)

def visualize(img):
    _min = img.min()
    _max = img.max()
    normalized_img = (img - _min)/ (_max - _min)
    return normalized_img


def main():
    args = create_argparser().parse_args()
    dist_util.setup_dist(args)
    logger.configure(dir = args.out_dir)
    tran_list = [transforms.ToTensor()]
    transform_test = transforms.Compose(tran_list)
    print("Your current directory : ", args.data_dir)
    ds = CustomDataset(args, args.data_dir, transform_test, mode='Test')
    args.in_ch = 7
    datal = th.utils.data.DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True
    )
    data = iter(datal)

    logger.log("creating model and diffusion...")

    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    all_images = []
    state_dict = dist_util.load_state_dict(args.model_path, map_location="cpu")
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        # name = k[7:] # remove `module.`
        if 'module.' in k:
            new_state_dict[k[7:]] = v
            # load params
        else:
            new_state_dict = state_dict

    model.load_state_dict(new_state_dict)

    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()
    num_tqdm = range(len(datal))
    # i_sample = 0
    count = 0
    for _ in tqdm(num_tqdm, desc='Processing'):
        if count >= 500:
            break
        b, m, path = next(data)  # should return an image from the dataloader "data"
        
        # Channel adaptation
        # Expected input to model (before noise cat) is 6 channels (based on args.in_ch=7)
        # b shape is [Batch, C, H, W]
        if args.modality == 'intensity':
            # Assume b is intensity (1 or 3 ch). 
            # We want to place it in the first 3 channels (or 1?) and zero the rest?
            # Assuming model expects [Intensity(3), Range(3)]
            if b.shape[1] == 1:
                b = b.repeat(1, 3, 1, 1) # to 3 channels
            elif b.shape[1] == 3:
                pass
            
            # Create zeros for range
            zeros = th.zeros_like(b)
            b = th.cat((b, zeros), dim=1) # 3+3=6
            
        elif args.modality == 'range':
            # Assume b is range (1 or 3 ch)
            if b.shape[1] == 1:
                b = b.repeat(1, 3, 1, 1)
            
            zeros = th.zeros_like(b)
            b = th.cat((zeros, b), dim=1) # 3+3=6
            
        elif args.modality == 'fused':
            # Expect b to be 6 channels? Or we might have to fuse manually if input is not 6 ch?
            # If the user provides 6-channel images in the '5d' folder, we are good.
            # If input is 1 or 3 channels, we can't really "fuse" without the other modality.
            # We assume for 'fused' mode, the input images are already 6 channels.
            if b.shape[1] < 6:
                # Fallback or error? 
                # For safety, if it's 2 channels (I, R), maybe repeat?
                # But let's assume correct data is provided for fused.
                pass
        
        c = th.randn_like(b[:, :1, ...])
        # i_sample += 1
        # if i_sample < 400:
        #     continue

        img = th.cat((b, c), dim=1)     # add a noise channel$
        slice_ID = path[0].split("/")[-1].split('.')[0]
        count += 1
        # print(slice_ID)
        logger.log(f"sampling {count}/500 : {slice_ID}...")
        start = th.cuda.Event(enable_timing=True)
        end = th.cuda.Event(enable_timing=True)
        enslist = []
        for i in range(args.num_ensemble):  # this is for the generation of an ensemble of 5 masks.
            model_kwargs = {}
            start.record()
            sample_fn = (
                diffusion.p_sample_loop_known if not args.use_ddim else diffusion.ddim_sample_loop_known
            )
            sample, x_noisy, org, cal, cal_out = sample_fn(
                model,
                (args.batch_size, 3, args.image_size, args.image_size), img,
                step = args.diffusion_steps,
                clip_denoised=args.clip_denoised,
                model_kwargs=model_kwargs,
            )
            end.record()
            th.cuda.synchronize()
            print('time for 1 sample', start.elapsed_time(end))  # time measurement for the generation of 1 sample
            # co = th.tensor(cal_out)
            co = cal_out.clone().detach()
            if args.version == 'new':
                enslist.append(sample[:,-1,:,:])
                # enslist.append(co[:,-1,:,:])
                # enslist.append(cal[:,-1,:,:])
            else:
                # enslist.append(co)
                enslist.append(sample[:, -1, :, :])
        x = staple(th.stack(enslist, dim=0)).squeeze(0)
        x = th.clamp(x, 0.0, 1.0)
        ensres = (x.mean(dim=0, keepdim=True).round())*255
        out_img = Image.fromarray((ensres[0].detach().cpu().numpy()).astype(np.uint8))
        out_img.save(os.path.join(args.out_dir, str(slice_ID)+'_output_ens'+".png"))
def create_argparser():
    defaults = dict(
        data_dir="./data/Test",
        clip_denoised=True,
        num_samples=1,
        batch_size=1,
        use_ddim=False,
        model_path="./pretrained_weights/savedmodel100000.pt",
        num_ensemble=1,
        gpu_dev="0",
        out_dir='./results/',
        multi_gpu=None,
        modality='fused', # intensity, range, fused
        debug=False
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":

    main()
