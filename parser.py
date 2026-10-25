import argparse

def get_parser():
    # MOST CRUCIAL DATA PARAMETERS
    parser = argparse.ArgumentParser(description='Stage1')
    parser.add_argument('--data_src', default='binned_dos_128',choices=['binned_dos_128', 'binned_dos_256', 'ph_dos_51', 'xanes_128_sigma0.5', 'xanes_O_128_sigma0.5', 'raman_128_sigma10'])

    # MOST CRUCIAL MODEL PARAMETERS
    parser.add_argument('--num_layers',default=3, type=int,
                        help='number of AGAT layers to use in model (default:3)')
    parser.add_argument('--num_heads',default=4, type=int,
                        help='number of Attention-Heads to use  per AGAT Layer (default:4)')
    parser.add_argument('--train_size',default=0.8, type=float, help='ratio size of the training-set (default:0.8)')
    parser.add_argument('--use_catached_data', default=True, type=bool)

    parser.add_argument("--train",action="store_true")  # default value is false
    parser.add_argument('--num-epochs',default=200, type=int)
    parser.add_argument('--c_epochs',default=20, type=int)
    parser.add_argument('--batch-size',default=256, type=int)
    parser.add_argument('--dec_layers',default=2, type=int)
    parser.add_argument('--dec_dropout',default=0.1, type=float)
    parser.add_argument('--temp',default=2., type=float)
    parser.add_argument('--clip',default=1., type=float)

    parser.add_argument('--accum_step', default=1, type=int)
    parser.add_argument('--lambda_factor', default=0.5, type=float)
    parser.add_argument('--lambda_scale', default=1.5, type=float)

    parser.add_argument('--d_model',default=512, type=int)
    parser.add_argument('--h',default=4, type=int)
    parser.add_argument('--d_ff',default=2048, type=int)
    parser.add_argument('-T0', "--T0", default=100, type=int, help='optimizer T0')
    parser.add_argument('--warmup',default=3000, type=int)
    parser.add_argument('--weight_decay',default=0., type=float)
    parser.add_argument('--rate_decay',default=2., type=float)

    parser.add_argument('--lr',default=0.001, type=float)
    parser.add_argument('--label_dim',default=128, type=int)
    parser.add_argument('--graph_dropout',default=0.1, type=float)
    parser.add_argument('--check-point-path', default=None, type=str)

    parser.add_argument('--MIR_lr', default=1e-3, type=float)
    parser.add_argument('--scaling_law', default=1.0, type=float)

    parser.add_argument('--seed', default=1, type=int)

    return parser

