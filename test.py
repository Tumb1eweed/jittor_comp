import os
import argparse
import torch
import numpy as np

from datasets import *
from utils.misc import seed_all, get_logger, str_list
from utils.transforms import NormalizeUnitSphere
from models.pgd import PGDModel
from models.utils import chamfer_distance_unit_sphere
from evaluate import Evaluator

def input_iter(input_dir):
    for fn in sorted(os.listdir(input_dir)):
        if fn[-3:] != 'xyz':
            continue
        pcl_noisy = torch.FloatTensor(np.loadtxt(os.path.join(input_dir, fn)))
        pcl_noisy, center, scale = NormalizeUnitSphere.normalize(pcl_noisy)
        yield {
            'pcl_noisy': pcl_noisy,
            'name': fn[:-4],
            'center': center,
            'scale': scale
        }

def main(noise):
    try:
        for resolution in args.resolutions:

            # Input/Output
            input_dir = os.path.join(args.input_root, '%s_%s_%s' % (args.dataset, resolution, noise))
            save_title = '{dataset}_Ours{modeltag}_{tag}_{res}_{noise}'.format_map({
                'dataset': args.dataset,
                'modeltag': '' if args.niters == 1 else '%dx' % args.niters,
                'tag': args.tag,
                'res': resolution,
                'noise': noise
            })
            output_dir = os.path.join(args.output_root, save_title)

            if not os.path.exists(output_dir):
                os.makedirs(output_dir)    # Output point clouds

            logger = get_logger('PGD_test_'+args.dataset+'_'+resolution+'_'+noise, output_dir)
            for k, v in vars(args).items():
                logger.info('[ARGS::%s] %s' % (k, repr(v)))

            device = args.device if torch.cuda.is_available() else 'cpu'
            if device == 'cuda':
                torch.cuda.set_per_process_memory_fraction(0.8)
                torch.cuda.empty_cache()
                
            logger.info(f'Using device: {device}')

            # Model
            try:
                model = PGDModel.load_from_checkpoint(args.ckpt, map_location=device)
                model = model.to(device)
                if args.sampling_timesteps is not None:
                    model.sampling_timesteps = int(args.sampling_timesteps)
                    if hasattr(model, 'args'):
                        model.args.sampling_timesteps = int(args.sampling_timesteps)
                    logger.info(f'Override sampling_timesteps to {model.sampling_timesteps}')
                logger.info('Model loaded successfully')
            except Exception as e:
                logger.error(f'Failed to load model: {str(e)}')
                return

            # Denoise
            for data in input_iter(input_dir):
                logger.info(data['name'])
                pcl_noisy = data['pcl_noisy'].to(device)
                with torch.no_grad():
                    model.eval()
                    pcl_next = pcl_noisy
                    for _ in range(args.niters):
                        pcl_next = model.patch_based_denoise(pcl_noisy=pcl_next,
                                                            patch_size=args.patch_size, 
                                                            seed_k=args.seed_k, 
                                                            seed_k_alpha=args.seed_k_alpha)
                    pcl_denoised = pcl_next.cpu()
                    # Denormalize
                    pcl_denoised = pcl_denoised * data['scale'] + data['center']
                
                save_path = os.path.join(output_dir, data['name'] + '.xyz')
                np.savetxt(save_path, pcl_denoised.numpy(), fmt='%.8f')

            if not args.dataset.startswith('RueMadame'): 
                # Evaluate
                evaluator = Evaluator(
                    output_pcl_dir=output_dir,
                    dataset_root=args.dataset_root,
                    dataset='PUNet' if args.dataset.startswith('PUNet') else args.dataset,
                    summary_dir=args.output_root,
                    experiment_name=save_title,
                    device=device,
                    res_gts=resolution,
                    logger=logger
                )
                evaluator.run()
    except Exception as e:
        print(f"Error in processing noise level {noise}: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':

    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, default=f'pretrained/PGD.ckpt')
    parser.add_argument('--input_root', type=str, default='./data/examples')
    parser.add_argument('--output_root', type=str, default='./data/results/PGD')
    parser.add_argument('--dataset_root', type=str, default='./data')
    parser.add_argument('--dataset', type=str, default='PUNet')
    parser.add_argument('--tag', type=str, default='')
    parser.add_argument('--resolutions', type=str_list, default=['10000_poisson', '50000_poisson']) # Set your test resolution
    parser.add_argument('--noise_lvls', type=str_list, default=['0.03']) # Set your test noise level
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=2025)

    # Filtering parameters
    parser.add_argument('--patch_size', type=int, default=1000)
    parser.add_argument('--niters', type=int, default=3)

    # Patch stitching params
    parser.add_argument('--seed_k', type=int, default=6)
    parser.add_argument('--seed_k_alpha', type=int, default=10)
    parser.add_argument('--sampling_timesteps', type=int, default=5)
    
    
    args = parser.parse_args()
    seed_all(args.seed)

    for noise in args.noise_lvls:
        main(noise)
