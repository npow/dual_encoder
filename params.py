import argparse
import pickle

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

argparser = argparse.ArgumentParser()
argparser.add_argument('--checkpoint_dir', type=str, default='.')
argparser.add_argument('--pretrained_model_dir', type=str)
argparser.add_argument('--pretrained_vectors', type=str, default='glove')
argparser.add_argument('--lr', type=float, default=0.001)
argparser.add_argument('--num_epochs', type=int, default=100)
argparser.add_argument('--batch_size', type=int, default=256)
argparser.add_argument('--hidden_size', type=int, default=200)
argparser.add_argument('--patience', type=int, default=0)
argparser.add_argument("--use_memory", type=str2bool, nargs='?', const=True, default=False)
argparser.add_argument("--force_use_memory", type=str2bool, nargs='?', const=True, default=False)
argparser.add_argument("--fine_tune_W", type=str2bool, nargs='?', const=True, default=True)
argparser.add_argument("--eval_only", type=str2bool, nargs='?', const=True, default=False)
argparser.add_argument("--debug", type=str2bool, nargs='?', const=True, default=False)
args = argparser.parse_args()

if args.eval_only:
    assert(args.pretrained_model_dir is not None)

if args.force_use_memory:
    assert(args.use_memory is not None)

with open('{}/params.pkl'.format(args.checkpoint_dir), 'wb') as f:
    pickle.dump(args.__dict__, f)
