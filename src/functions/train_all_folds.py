import subprocess
import time
import argparse
import datetime

if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate nn-Unet results')
    parser.add_argument("--nnUNet_trainer", help='nnUNet trainer used for training', type=str, default='nnUNetTrainer')
    parser.add_argument("--nnUNet_plans", help='nnUNet plans used for training', type=str, default='nnUNetPlans')
    parser.add_argument("--configuration", help='Configuration used for training', type=str, default='3d_fullres')
    parser.add_argument("--dataset", help='Dataset name or Dataset number', required=True, type=str)
    parser.add_argument("--start_fold", help='starting fold', type=int, default=0)
    
    args = parser.parse_args()

    for fold in range(args.start_fold, 5):
        print(fold)
        start_time = time.time()
        subprocess.run(['nnUNetv2_train', '-tr', args.nnUNet_trainer, '-p', args.nnUNet_plans,
                         args.dataset, args.configuration, str(fold)])
        print(str(datetime.timedelta(seconds=int(time.time() - start_time))))