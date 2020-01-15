import argparse
import os

def get_args():

    # save_im_path = "./g_z/" + run_name
    # if n_gpu == 1 or n_gpu == 2:
    #     save_checkpoints_path = "./checkpoints/" + run_name
    # elif n_gpu == 4:
    #     save_checkpoints_path = "/hpf/largeprojects/agoldenb/lechang/" + run_name
    #
    # if n_gpu == 1 or n_gpu == 2:
    #     ae_dir = "./ae_checkpoints/ae-9600.pth"
    # else:
    #     ae_dir = "./ae-9600.pth"

    # if args.n_gpu == 1:
    #     batch_size = {(25, 8): 64, (50, 16): 32, (100, 32): 2, (200, 64): 2, (400, 128): 2, (800, 256): 2}
    # if args.n_gpu == 2:
    #     batch_size = {(25, 8): 512, (50, 16): 512, (100, 32): 24, (200, 64): 16, (400, 128): 4, (800, 256): 4}
    # elif args.n_gpu == 4:
    #     batch_size = {(25, 8): 512, (50, 16): 360, (100, 32): 360, (200, 64): 72, (400, 128): 32, (800, 256): 6}


    # batch_size = {(25, 8): 64, (50, 16): 32, (100, 32): 2, (200, 64): 2, (400, 128): 2, (800, 256): 2}
    batch_size="360,180,90,36,2,2"


    run_name = "SR GAN"
    save_im_path = "./g_z/" + run_name
    save_checkpoints_path = "./checkpoints/" + run_name

    # load_checkpoint = "/hpf/largeprojects/agoldenb/lechang/checkpoints/Instance noise 0.2/trained-51000.pth"


    parser = argparse.ArgumentParser()
    parser.add_argument('--n_gpu', type=int, default=1, metavar='N',
                        help='')
    parser.add_argument('--g_steps', type=int, default=1, metavar='N',
                        help='')
    parser.add_argument('--run_name', type=str, default="SR GAN", metavar='N',
                        help='')
    parser.add_argument('--batch_size', type=str, default=batch_size, metavar='N',
                         help='')
    parser.add_argument('--lr', type=float, default=0.001, metavar='N',
                         help='')
    parser.add_argument('--data_path', type=str, default="./data", metavar='N',
                         help='')
    parser.add_argument('--g_z_path', type=str, default="./g_z/", metavar='N',
                         help='')
    parser.add_argument('--save_checkpoints', type=str, default="./checkpoints/", metavar='N',
                         help='')
    parser.add_argument('--load_checkpoint', type=str, default="no", metavar='N',
                         help='')
    parser.add_argument('--lambda1', type=str, default=0.01, metavar='N',
                         help='')
    parser.add_argument('--lambda2', type=str, default=0.01, metavar='N',
                         help='')
    parser.add_argument('--wandb_dir', type=str, default="./wandb", metavar='N',
                         help='')
    parser.add_argument('--n_sample', type=int, default=600_000, metavar='N',
                        help='')
    parser.add_argument('--spectral_reg', action='store_true')
    parser.add_argument('--instance_noise', type=float, default=0, metavar='N',
                        help='')
    parser.add_argument('--ae_dir', type=str, default="./ae_checkpoints/ae-9600.pth", metavar='N',
                        help='')

    args = parser.parse_args()

    os.makedirs(args.g_z_path + args.run_name, exist_ok=True)
    os.makedirs(args.save_checkpoints + args.run_name, exist_ok=True)

    if args.n_gpu == 1:
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    if args.n_gpu == 2:
        os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'
    if args.n_gpu == 4:
        os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2, 3'

    return args