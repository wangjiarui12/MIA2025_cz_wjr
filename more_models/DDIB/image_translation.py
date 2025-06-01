"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""
import sys
sys.path.append('../')

from guided_diffusion.image_datasets import ImageDataset, list_image_files
from torch.utils.data import DataLoader, Dataset

import argparse
import os

import numpy as np
import torch as th
import torch.distributed as dist
from PIL import Image
from mpi4py import MPI

from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)

def load_source_data_for_domain_translation(
        *,
        image_size,
        dataA_dir="./experiments/imagenet",
):
    """
    This function is new in DDIBs: loads the source dataset for translation.
    For the dataset, create a generator over (images, kwargs) pairs.
    No image cropping, flipping or shuffling.

    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.
    """
    if not dataA_dir:
        raise ValueError("unspecified data directory")
    all_files = [f for f in list_image_files(dataA_dir)]
    print(all_files)
    # Classes are the first part of the filename, before an underscore: e.g. "291_1.png"
    classes = None
    dataset = ImageDataset(
        image_size,
        all_files,
        random_flip=False,
        classes=classes,
        filepaths=all_files,
        shard=MPI.COMM_WORLD.Get_rank(),
        num_shards=MPI.COMM_WORLD.Get_size()
    )
    loader = DataLoader(dataset, batch_size=dataset.__len__(), shuffle=False, num_workers=1)
    yield from loader

def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist(args.single_gpu)
    logger.configure(args.log_root)

    logger.log("creating model and diffusion...")
    modelA, diffusionA = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    modelA.load_state_dict(
        dist_util.load_state_dict(args.modelA_path, map_location="cpu")
    )
    modelA.to(dist_util.dev())
    if args.use_fp16:
        modelA.convert_to_fp16()
    modelA.eval()

    modelB, diffusionB = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    modelB.load_state_dict(
        dist_util.load_state_dict(args.modelB_path, map_location="cpu")
    )
    modelB.to(dist_util.dev())
    if args.use_fp16:
        modelB.convert_to_fp16()
    modelB.eval()

    data = load_source_data_for_domain_translation(
        image_size=args.image_size,
        dataA_dir=args.dataA_dir
    )
    with th.no_grad():
        for i,(batch,extra) in enumerate(data):
            batch = batch.to(dist_util.dev())
            # First, use DDIM to encode to latents.
            logger.log("encoding the source images.")
            # z = diffusionA.encode(
            #     model=modelA,
            #     image=batch,
            #     device=dist_util.dev(),
            # )

            z = diffusionA.ddim_reverse_sample_loop(
                model=modelA,
                image=batch,
                clip_denoised=False,
                device=dist_util.dev()
                )
            logger.log(f"obtained latent representation for {batch.shape[0]} samples...")
            logger.log(f"latent with mean {z.mean()} and std {z.std()}")
        
            # Next, decode the latents to the target class.
            # sample,x_T = diffusionB.generate(
            #     model=modelB,
            #     z=z,
            #     device=dist_util.dev(),
            # )

            # xt可以正常生成

            sample = diffusionB.ddim_sample_loop(
                model=modelB,
                shape=[batch.shape[0],3,256,256],
                noise=z,
                clip_denoised=True,
                device=dist_util.dev()
                )
            
            sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
            sample = sample.permute(0, 2, 3, 1)
            sample = sample.contiguous()
 
            
            
            

            images = []
            gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
            images.extend([sample.cpu().numpy() for sample in gathered_samples])
            logger.log(f"created {len(images) * args.batch_size} samples")

            logger.log("saving translated images.")
            images = np.concatenate(images, axis=0)

            for index in range(images.shape[0]):
                base_dir, filename = os.path.split(extra["filepath"][index])
                base_dir = os.path.join(args.log_root+'/images/')
                if not os.path.exists(base_dir):
                    os.makedirs(base_dir)  # 创建文件夹
                filename, ext = filename.split(".")
                filepath = os.path.join(base_dir, f"{filename}_translated.{ext}")
                image = Image.fromarray(images[index])
                image.save(filepath)
                logger.log(f"    saving: {filepath}")

    dist.barrier()
    logger.log(f"domain translation complete")


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=16,
        batch_size=1,
        use_ddim=False,
        single_gpu=True,
        model_path="",
        log_root="./translated_images",
        get_images=True,
        dataA_dir='test_set',
        modelA_path='checkpoints/LQ_050000.pt',
        modelB_path='checkpoints/HQ_050000.pt',
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()