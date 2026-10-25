import random
import sys
from copy import deepcopy

import numpy as np
import torch
from torch_geometric.data import DataLoader as torch_DataLoader
from tqdm import tqdm

from parser import get_parser
from Sci_Reasoner.data import CIF_Dataset, CIF_Lister, load_benchmark
from Sci_Reasoner.stage1 import Structure_Aware_Spectrum_Decoding
from Sci_Reasoner.stage2 import MIR
from Sci_Reasoner.utils import metrics


def test(args):
    torch.autograd.set_detect_anomaly(True)
    device = torch.device(f'cuda:0')


    # DATA PARAMETERS
    random_num = args.seed
    random.seed(random_num)
    np.random.seed(random_num)
    torch.manual_seed(random_num)
    torch.cuda.manual_seed(random_num)
    # MODEL HYPER-PARAMETERS
    batch_size      = args.batch_size
    valid_param     = {'batch_size':batch_size, 'shuffle': False}

    # DATALOADER/ TARGET NORMALIZATION
    pd_data, train_idx, val_idx, test_idx = load_benchmark(args.data_src)
    CRYSTAL_DATA = CIF_Dataset(args, pd_data=pd_data)


    ckpt_dir_stage1 = args.stage1_ckpt_1
    ckpt_dir_stage2 = args.stage2_ckpt_1
    print("model:", ckpt_dir_stage1)
    print("model:", ckpt_dir_stage2)
    ckpt_dir_stage3 = args.stage1_ckpt_2
    ckpt_dir_stage4 = args.stage2_ckpt_2
    print("model:", ckpt_dir_stage3)
    print("model:", ckpt_dir_stage4)

    if isinstance(train_idx, np.ndarray):
        train_idx = train_idx.tolist()
        val_idx = val_idx.tolist()
        test_idx = test_idx.tolist()

    print('testing size:', len(test_idx))

    # Modified
    testing_set = CIF_Lister(test_idx, CRYSTAL_DATA, df=pd_data)

    print(f'> USING MODEL Sci_Reasoner!')
    # MODEL
    net_base_1 = Structure_Aware_Spectrum_Decoding(args).to(device)
    net_1 = MIR(args).to(device)
    check_point_path = './TRAINED/stage1/' + ckpt_dir_stage1 + '/model.ckpt'
    check_point = torch.load(check_point_path, weights_only=False)
    state_dict = check_point['model']  
    net_base_1.load_state_dict(state_dict, strict=False)
    check_point_path = './TRAINED/stage2/' + ckpt_dir_stage2 + '/model.ckpt'
    check_point = torch.load(check_point_path, weights_only=False)
    state_dict = check_point['model']
    net_1.load_state_dict(state_dict, strict=False)

    net_base_2 = Structure_Aware_Spectrum_Decoding(args).to(device)
    net_2 = MIR(args).to(device)
    check_point_path = './TRAINED/stage1/' + ckpt_dir_stage3 + '/model.ckpt'
    check_point = torch.load(check_point_path, weights_only=False)
    state_dict = check_point['model']  
    net_base_2.load_state_dict(state_dict, strict=False)
    check_point_path = './TRAINED/stage2/' + ckpt_dir_stage4 + '/model.ckpt'
    check_point = torch.load(check_point_path, weights_only=False)
    state_dict = check_point['model']
    net_2.load_state_dict(state_dict, strict=False)


    print(f'> TESTING MODEL ...')
    test_loader   = torch_DataLoader(dataset=testing_set, **valid_param)


    prediction = []
    label_gt = []
    # TESTING-PHASE
    net_1.eval()
    net_base_1.eval()
    net_2.eval()
    net_base_2.eval()
    for data in tqdm(test_loader, mininterval=0.5, desc='(testing)', position=0, leave=True, ascii=True):
        data = data.to(device)
        if isinstance(data.y, tuple) or isinstance(data.y, list):
            data.y = data.y[0]
        valid_label = deepcopy(data.y).float().to(device)

        with torch.no_grad():
            crystal_ctx, _, _ = net_base_1.graph_encoder(data)
            p0 = net_base_1(data)
            p0 = p0 / torch.sum(p0, dim=1, keepdim=True)
            pT_pred_1 = net_1.inference(p0.to(device), crystal_ctx, num_steps=args.step)

            crystal_ctx, _, _ = net_base_2.graph_encoder(data)
            p0 = net_base_2(data)
            p0 = p0 / torch.sum(p0, dim=1, keepdim=True)
            pT_pred_2 = net_2.inference(p0.to(device), crystal_ctx, num_steps=args.step)
        
        pT_pred_final = 0.5*pT_pred_1 + 0.5*pT_pred_2

        valid_label_sum = torch.sum(valid_label, dim=1, keepdim=False)
        prediction.append((pT_pred_final*valid_label_sum.unsqueeze(-1)).detach().cpu().numpy())
        label_gt.append(valid_label.detach().cpu().numpy())


    prediction = np.concatenate(prediction, axis=0)
    label_gt = np.concatenate(label_gt, axis=0)
    

    mae, r2, mse, wd = metrics(label_gt, prediction)
    print(f"\n********** Result ***********")
    print("r2 = %.4f" % r2)
    print("mae = %.4f" % mae)
    print("mse = %.4f" % mse)
    print("wd = %.4f" % wd)
    print("\n*****************************************")



def main():
    parser = get_parser()
    parser.add_argument("--stage1_ckpt_1", required=True,
                        help="The first Stage-1 checkpoint dir name under TRAINED/stage1/")
    parser.add_argument("--stage2_ckpt_1", required=True,
                        help="The first Stage-2 checkpoint dir name under TRAINED/stage2/")
    parser.add_argument("--stage1_ckpt_2", required=True,
                        help="The second Stage-1 checkpoint dir name under TRAINED/stage1/")
    parser.add_argument("--stage2_ckpt_2", required=True,
                        help="The second Stage-2 checkpoint dir name under TRAINED/stage2/")
    parser.add_argument("--step", default=100, type=int,
                        help="Number of MIR refinement steps (default 100)")

    args = parser.parse_args(sys.argv[1:])
    test(args)

if __name__ == '__main__':
    main()
