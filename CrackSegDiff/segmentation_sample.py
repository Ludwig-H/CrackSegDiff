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
    args.in_ch = 10
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
    pbar = tqdm(num_tqdm, desc='Processing')
    for _ in pbar:
        if count >= 500:
            break
        b, m, path = next(data)  # should return an image from the dataloader "data"
        
        # Channel adaptation logic
        # We need to construct a 9-channel data tensor from b
        # b might be 3 channels (single modality file) or 6/9 channels (merged file)
        
        b_final = None
        
        if args.modality == 'intensity':
            # Intensity is channels 0-3
            if b.shape[1] >= 3:
                part = b[:, :3, :, :]
            elif b.shape[1] == 1:
                part = b.repeat(1, 3, 1, 1)
            else:
                part = th.zeros((b.shape[0], 3, b.shape[2], b.shape[3]), device=b.device)
            
            # Pad with 6 zeros (for range and filtered)
            zeros = th.zeros((part.shape[0], 6, part.shape[2], part.shape[3]), device=part.device)
            b_final = th.cat((part, zeros), dim=1)
            
        elif args.modality == 'range':
            # Range is channels 3-6
            # If input is large (>=6), assume range is at 3:6
            # If input is small (3), assume it IS the range image (so 0:3)
            if b.shape[1] >= 6:
                part = b[:, 3:6, :, :]
            else:
                part = b[:, :3, :, :]
                
            if part.shape[1] == 1: part = part.repeat(1, 3, 1, 1)
            
            zeros_pre = th.zeros((part.shape[0], 3, part.shape[2], part.shape[3]), device=part.device)
            zeros_post = th.zeros((part.shape[0], 3, part.shape[2], part.shape[3]), device=part.device)
            b_final = th.cat((zeros_pre, part, zeros_post), dim=1)
            
        elif args.modality == 'filtered':
            # Filtered is channels 6-9
            # If input is large (>=9), assume filtered is at 6:9
            # If input is small (3), assume it IS the filtered image
            if b.shape[1] >= 9:
                part = b[:, 6:9, :, :]
            else:
                part = b[:, :3, :, :]
            
            if part.shape[1] == 1: part = part.repeat(1, 3, 1, 1)
            
            zeros = th.zeros((part.shape[0], 6, part.shape[2], part.shape[3]), device=part.device)
            b_final = th.cat((zeros, part), dim=1)
            
        elif args.modality == 'fused':
            # Fused usually means Intensity + Range (channels 0-6)
            # We keep it as is, padding the rest (filtered) with zeros
            
            part_int = None
            part_rng = None
            
            # Extract Intensity
            if b.shape[1] >= 3:
                part_int = b[:, :3, :, :]
            elif b.shape[1] == 1:
                part_int = b.repeat(1,3,1,1)
            
            # Extract Range
            if b.shape[1] >= 6:
                part_rng = b[:, 3:6, :, :]
            elif b.shape[1] == 3:
                # Ambiguous if we only have 3 channels. 
                # If we are in 'fused' mode but only have 3 channels, 
                # maybe we lack range? Or maybe the 3 channels ARE fused?
                # Let's assume we lack range and pad with zeros, or fail?
                # To be safe, let's pad with zeros if missing.
                part_rng = th.zeros_like(part_int)
            else:
                part_rng = th.zeros_like(part_int) # fallback
                
            # Pad Filtered (3 zeros)
            zeros_filt = th.zeros((part_int.shape[0], 3, part_int.shape[2], part_int.shape[3]), device=part_int.device)
            
            b_final = th.cat((part_int, part_rng, zeros_filt), dim=1)
            
        elif args.modality == 'all':
            # Expecting 9 channels for full multi-modal inference
            if b.shape[1] >= 9:
                b_final = b[:, :9, :, :]
            else:
                # Pad whatever we have to 9 channels
                print(f"Warning: 'all' modality expects 9 channels, but got {b.shape[1]}. Padding with zeros.")
                pad_size = 9 - b.shape[1]
                if pad_size > 0:
                    zeros = th.zeros((b.shape[0], pad_size, b.shape[2], b.shape[3]), device=b.device)
                    b_final = th.cat((b, zeros), dim=1)
                else:
                    b_final = b
        
        else:
             # Default fallback if unknown modality
             print(f"Unknown modality {args.modality}, using raw input padded/sliced to 9.")
             if b.shape[1] >= 9:
                 b_final = b[:, :9, :, :]
             else:
                 pad_size = 9 - b.shape[1]
                 if pad_size > 0:
                     zeros = th.zeros((b.shape[0], pad_size, b.shape[2], b.shape[3]), device=b.device)
                     b_final = th.cat((b, zeros), dim=1)
                 else:
                     b_final = b[:, :9, :, :]

        b = b_final

        c = th.randn_like(b[:, :1, ...])
        # i_sample += 1
        # if i_sample < 400:
        #     continue

        img = th.cat((b, c), dim=1)     # add a noise channel$
        slice_ID = path[0].split("/")[-1].split('.')[0]
        count += 1
        pbar.set_description(f"Processing {count}/500: {slice_ID}")
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
