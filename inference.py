import torch
import torch.backends.cudnn as cudnn
import numpy as np
from typing import Iterable
from tqdm import tqdm
import argparse

import timm
import timm.optim.optim_factory as optim_factory

import util.misc as misc
from llama import Tokenizer
from llama_vqa import LLaMA_VQA
from dataloader import load_data

def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int, help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=400, type=int)
    parser.add_argument('--accum_iter', default=1, type=int, help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--llama_model_path', default='./pretrained/llama/', type=str, help='path of llama model')
    parser.add_argument('--model', default='llama7B_adapter', type=str, metavar='MODEL', help='Name of model to train')
    parser.add_argument('--adapter_layer', type=int, default=32, metavar='LENGTH', help='the number of adapter layer')
    parser.add_argument('--adapter_len', type=int, default=10, metavar='LENGTH', help='the adapter length')
    parser.add_argument('--max_seq_len', type=int, default=512, metavar='LENGTH', help='the maximum sequence length')
    parser.add_argument('--max_feats', type=int, default=10, metavar='LENGTH', help='the maximum feature length')

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05, help='weight decay (default: 0.05)')
    parser.add_argument('--lr', type=float, default=None, metavar='LR', help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR', help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR', help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N', help='epochs to warmup LR')

    # Dataset parameters
    parser.add_argument('--dataset', default='nextqa', type=str, help='dataset')
    parser.add_argument('--output_dir', default='./output_dir', help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='start epoch')
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--pin_mem', action='store_true', help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    
    parser.add_argument('--vaq', action='store_true', help='vaq loss')
    parser.add_argument('--qav', action='store_true', help='qav loss')
    parser.add_argument('--bias', type=float, default=3., help='attention bias')
    parser.add_argument('--tau', type=float, default=100., help='tau')
    parser.add_argument('--sub', action='store_true', help='subtitles for VLEP and TVQA')

    parser.add_argument('--eval', type=bool, default=True)
    return parser

def test(model: torch.nn.Module, data_loader: Iterable, args=None):
    model.eval()
    total_correct = 0
    total_samples = 0

    num_categories = 4  # set as per dataset
    total_cat_cnts = [0] * num_categories
    total_corr_cnts = [0] * num_categories

    qtype_mapping = {'descriptive':1, 'explanatory':2, 'predictive':3, 'counterfactual':4}

    fm = open(mistakes_file, 'w')

    for data_iter_step, data in enumerate(data_loader):
        qtypes = data['qtype']
        answer = data['answer'].cuda()
        bsz = answer.shape[0]

        with torch.no_grad():
            logits = model(data, inference=True)
        
        count = (logits != 0).sum(-1)
        prediction = (logits.sum(-1) / count).argmin(-1)

        eval = (answer == prediction)

        for i in range(len(data['answer'])):
            qn_type = qtypes[i]
            total_cat_cnts[qn_type-1] += 1
            if answer[i] == prediction[i]:
                total_corr_cnts[qn_type-1] += 1
            else:
                fm.write(data[''])

        num_correct = eval.sum().item()
        total_samples += bsz
        total_correct += num_correct

    test_acc = total_correct / total_samples
    cat_acc = {qtype: (total_corr_cnts[qtype_mapping[qtype]-1] / total_cat_cnts[qtype_mapping[qtype]-1]) for qtype in qtype_mapping}
    return test_acc, cat_acc

def main(args, mistakes_file):
    misc.init_distributed_mode(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    tokenizer = Tokenizer(model_path=f'{args.llama_model_path}./tokenizer.model')

    data_loader_test = load_data(args, tokenizer, split='test')

    model = LLaMA_VQA(args)
    model.to(device)

    model_without_ddp = model

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256
    
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
    
    # following timm: set wd as 0 for bias and norm layers
    param_groups = optim_factory.param_groups_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=None)

    test_acc, cat_acc = test(model, data_loader_test, args, mistakes_file)

    print("Test accuracy :", test_acc)
    print("Categorical test accuracy : ", cat_acc)

if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    print("Resume :", args.resume)
    print("Eval :", args.eval)
    mistakes_file = "mistakes.txt"
    main(args, mistakes_file)

