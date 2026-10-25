import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import DataLoader as torch_DataLoader
from tqdm import tqdm

from parser import get_parser
from Sci_Reasoner.data import CIF_Dataset, CIF_Lister, load_benchmark, mkdirs
from Sci_Reasoner.stage1 import Structure_Aware_Spectrum_Decoding, compute_loss
from Sci_Reasoner.transformer import rate
from Sci_Reasoner.utils import metrics

best_valid_loss=1e+10


def train_model(args):
    ckpt_dir = args.save_dir
    log_dir = './TRAINED/stage1/' + ckpt_dir
    mkdirs(log_dir)
    log_file = open(log_dir + '/log.txt', 'w')

    def print_log(msg):
        print(msg)
        print(msg, file=log_file)


    random_num = args.seed
    random.seed(random_num)
    np.random.seed(random_num)
    torch.manual_seed(random_num)
    torch.cuda.manual_seed(random_num)
    torch.backends.cudnn.deterministic = True

    # torch.autograd.set_detect_anomaly(True)
    torch.cuda.set_device(0)
    torch.backends.cudnn.benchmark = False
    device = torch.device('cuda:0')


    # MODEL HYPER-PARAMETERS
    num_epochs      = args.num_epochs
    learning_rate   = args.lr
    batch_size      = args.batch_size


    # DATALOADER/ TARGET NORMALIZATION
    pd_data, train_idx, val_idx, test_idx = load_benchmark(args.data_src)
    CRYSTAL_DATA = CIF_Dataset(args, pd_data=pd_data)

    if isinstance(train_idx, np.ndarray):
        train_idx = train_idx.tolist()
        val_idx = val_idx.tolist()
        test_idx = test_idx.tolist()

    if args.scaling_law < 1.0:
        sample_size = int(len(train_idx) * args.scaling_law)
        train_idx = random.sample(train_idx, sample_size)

    
    print_log(f'training size: {len(train_idx)}, min/max: {min(train_idx)} {max(train_idx)}')
    print_log(f'validation size: {len(val_idx)}, min/max: {min(val_idx)} {max(val_idx)}')
    print_log(f'testing size: {len(test_idx)}, min/max:, {min(test_idx)}, {max(test_idx)}')
    print_log(f'total size: {len(train_idx)}, {len(val_idx)+len(test_idx)}')


    training_set       =  CIF_Lister(train_idx,CRYSTAL_DATA,df=pd_data)
    validation_set     =  CIF_Lister(val_idx,CRYSTAL_DATA,df=pd_data)
    testing_set        =  CIF_Lister(test_idx, CRYSTAL_DATA, df=pd_data)

    print_log(f'> USING MODEL Structure_Aware_Spectrum_Decoding!')
    net = Structure_Aware_Spectrum_Decoding(args).to(device)

    def count_params(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    print_log(f"model_size: {count_params(net)}")

    
    # LOSS & OPTMIZER & SCHEDULER
    optimizer = optim.Adam(net.parameters(), lr=learning_rate, betas=(0.9, 0.98), eps=1e-9, weight_decay=args.weight_decay)
    shift = args.warmup - args.warmup**(1./args.lambda_scale)
    one_epoch_steps = len(train_idx) // batch_size + 1

    print_log(f"## shift: {shift}")
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer=optimizer,
        lr_lambda=lambda step: rate(
            step, args.d_model, factor=args.lambda_factor, 
            warmup=args.warmup, scale=args.lambda_scale, 
            shift=shift, c_steps=one_epoch_steps * args.c_epochs,
            rate_decay = args.rate_decay
        ),
    )


    loss_fn = nn.KLDivLoss(reduction="batchmean", log_target=True).cuda()

    print_log(f'> TRAINING MODEL ...')
    train_loader   = torch_DataLoader(dataset=training_set,
                                      batch_size=batch_size,
                                      shuffle=True,
                                      num_workers=16,
                                      pin_memory=True
                                     )
    valid_loader   = torch_DataLoader(dataset=validation_set,
                                      batch_size=batch_size,
                                      shuffle=False,
                                      pin_memory=True
                                     )
    test_loader    = torch_DataLoader(dataset=testing_set,
                                      batch_size=batch_size,
                                      shuffle=False,
                                      pin_memory=True
                                     )

    total_loss = 0
    training_counter=0
    cur_step = 0
    current_step = 0


    summary_dir = './summary/' + ckpt_dir
    mkdirs(summary_dir)
    writer = SummaryWriter(log_dir=summary_dir)
    
    log_file.flush()

    start_time = time.time()
    for epoch in range(num_epochs):
        # TRAINING-STAGE
        net.train()
        args.train = True
        n_accum = 0

        for data in tqdm(train_loader, mininterval=0.5, desc=f'(EPOCH:{epoch} TRAINING)', position=0, leave=True, ascii=True):
            n_accum += 1
            current_step += 1
            data = data.to(device, non_blocking=True)
            if isinstance(data.y, tuple) or isinstance(data.y, list):
                data.y = data.y[0]
            train_label = data.y.to(device)
            
            train_label_sum = torch.sum(train_label, dim=1, keepdim=False)

            pred_logits = net(data)
            loss, pred = compute_loss(train_label, train_label_sum, pred_logits, args)
            
            loss /= args.accum_step
            loss.backward()
            
            if n_accum % args.accum_step == 0:
                nn.utils.clip_grad_norm_(net.parameters(), args.clip)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            if epoch < args.T0-1:
                scheduler.step()
 
            total_loss += loss
            training_counter +=1
            cur_step += 1
 
        
        optimizer.zero_grad(set_to_none=True)
        training_counter = 0
        total_loss = 0

        def valid_test(data_loader, net, mode="Valid"):
            global best_valid_loss

            # VALIDATION-PHASE
            valid_counter = 0
            total_loss = 0
            prediction = []
            label_gt = []

            net.eval()

            for data in tqdm(data_loader, mininterval=0.5, desc=f'({mode})', position=0, leave=True, ascii=True):
                data = data.to(device, non_blocking=True)
                if isinstance(data.y, tuple) or isinstance(data.y, list):
                    data.y = data.y[0]
                valid_label = data.y.float().to(device)

                valid_label_sum = torch.sum(valid_label, dim=1, keepdim=False)    

                
                with torch.no_grad():
                    pred_logits = net(data)
                    loss, pred = compute_loss(valid_label, valid_label_sum, pred_logits, args)

                prediction.append(pred.detach().cpu().numpy())
                label_gt.append(valid_label.detach().cpu().numpy())

                total_loss += loss
                valid_counter += 1

            avg_loss = total_loss / valid_counter

            prediction = np.concatenate(prediction, axis=0)
            label_gt = np.concatenate(label_gt, axis=0)

            mae, r2, mse, wd = metrics(label_gt, prediction)
            
            lr = optimizer.param_groups[0]['lr']
            writer.add_scalar(f"{mode}/lr", lr, epoch)
            writer.add_scalar(f"{mode}/loss", avg_loss, epoch)
            writer.add_scalar(f"{mode}/r2", r2, epoch)
            writer.add_scalar(f"{mode}/mae", mae, epoch)
            writer.add_scalar(f"{mode}/mse", mse, epoch)
            writer.add_scalar(f"{mode}/wd", wd, epoch)

            print_log(f"\n********** {epoch} {mode} STATISTIC ***********")
            print_log("lr = %.4f" % lr)
            print_log("\n")


            if mode == "Valid" and best_valid_loss > mse:
                best_valid_loss = mse
                print_log("\n********** SAVING MODEL ***********")
                checkpoint = {'model': net.state_dict(), 'args': args}
                save_path = './TRAINED/stage1/' + ckpt_dir
                mkdirs(save_path)
                save_path = save_path + '/model.ckpt'
                torch.save(checkpoint, save_path)
                print_log("A new model has been saved to " + save_path)
                print_log("\n*****************************************")

            valid_counter = 0
            total_loss = 0
            prediction = []
            label_gt = []

        valid_test(valid_loader, net, mode="Valid")
        log_file.flush()

    end_time = time.time()
    e_time = end_time - start_time
    print_log(f"> DONE TRAINING !")

    log_file.close()

def main():
    parser = get_parser()
    parser.add_argument("--save_dir", required=True,
                        help="Save best-val Stage-1 checkpoint under TRAINED/stage1/<save_dir>/")
    args = parser.parse_args(sys.argv[1:])
    train_model(args)

if __name__ == '__main__':
    main()
