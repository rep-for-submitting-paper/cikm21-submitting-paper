"""
train script
"""
import argparse
import torch
import random
import numpy as np
from setting import logger
from utils.load_data import load_data
from model_zoo.sage import GraphSageHandler
from model_zoo.han_sage import HANSageHander


parser = argparse.ArgumentParser(description="malware detection.")
parser.add_argument('--model',type=str, default="GraphSage", help='choose model:\n SupervisedGraphSage, UnsupervisedGraphSage, MultiGraphSage, MLP')
parser.add_argument('--view', type=str, default="app_permission_app", help="choose whick view will be loaded. (tpl or permission or multi)")
parser.add_argument('--label_rate', type=float, default=1, help="the rate of labeled samples in train set")

parser.add_argument('--dropout', type=float, default=0.4, help="p for dropout layer")
parser.add_argument('--embed_dim', type=int, default=200, help="the dim of embedding vector")
parser.add_argument('--num_sample', type=int, default=5, help="used in sage_sample_layer")

parser.add_argument('--epoches', type=int, default=5, help="train epoches")
parser.add_argument('--interval_eval', type=int, default=100, help="evaluation interval")
parser.add_argument('--batch_size', type=int, default=128, help="minibatch")
parser.add_argument('--seed', type=int, default=0, help="seed for random")
parser.add_argument('--freeze', type=bool, default=False, help="used in hansage to load pretrain sage")

parser.add_argument('--keyword', type=str, default="drebin", help="chose dense feat: drebin or mamadroid")

args = parser.parse_args()
args.cuda = torch.cuda.is_available()

if args.view == 'multi':
    args.model == 'HANSage'
    args.view = None
else:
    args.model == 'GraphSage'

if args.model == 'HANSage':
    args.view = None

SEED = 2020
random.seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)
if args.cuda:
    torch.cuda.manual_seed(SEED)
    # torch.cuda.set_device(setting.cuda_device_id)

logger.info(args)


def run(data_dir, model_path, embed_path, report_file):
    logger.info("loading dataset.")
    num_class = 2
    adj_lists, feat_data, labels = load_data(data_dir, args.keyword, args.view)
    data = dict()
    data['feat_data'] = feat_data
    data['labels'] = labels

    if args.model == "GraphSage":
        data['adj_lists'] = adj_lists[0]
        model = GraphSageHandler(num_class, data, args)
    elif args.model == "HANSage":
        data['adj_lists'] = adj_lists
        model = HANSageHander(num_class, data, args)
    else:
        raise("error")
    ret_tuple, df = model.train_ddc(epoch=args.epoches, interval_val=args.interval_eval)

    df.to_csv("{}/rret/{}_{}.csv".format(data_dir, args.keyword, args.view), index=False)
    

    import pickle as pkl
    with open('ret_tuple_{}.pkl'.format(args.view), 'wb') as f:
        pkl.dump(ret_tuple, f)
    model.save_mode(model_path, report_file)
    

if __name__ == '__main__':
    from setting import data_dir, embed_path, report_file, model_path
    run(data_dir=data_dir, model_path=model_path, embed_path=embed_path, report_file=report_file)
