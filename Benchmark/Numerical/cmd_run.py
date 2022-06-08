from cfg.option import Options
import os
from module.train import Trainer
import argparse
from random import randrange


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1', 'True'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0', 'False'):
        return False


def get_args():
    args = argparse.ArgumentParser()
    args.add_argument('-cfg', '--config_file',
                      default='cfg/CT.cfg', type=str,
                      help="Configure file containing parameters for the algorithm")
    args.add_argument('-s', '--save_path', default='results/test',
                      type=str)
    args.add_argument('--model', default='Autoformer',
                      type=str)
    args.add_argument('-mr', '--multi_run', default=1,
                      type=int)
    args.add_argument('-se', '--seed', default=0,
                      type=int)
    # model define
    args.add_argument('--bucket_size', type=int, default=4, help='for Reformer')
    args.add_argument('--n_hashes', type=int, default=4, help='for Reformer')
    args.add_argument('--enc_in', type=int, default=1, help='encoder input size')
    args.add_argument('--dec_in', type=int, default=1, help='decoder input size')
    args.add_argument('--c_out', type=int, default=1, help='output size')
    args.add_argument('--d_model', type=int, default=512, help='dimension of model')
    args.add_argument('--n_heads', type=int, default=8, help='num of heads')
    args.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    args.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    args.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    args.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    args.add_argument('--factor', type=int, default=1, help='attn factor')
    args.add_argument('--distil', action='store_false',
                      help='whether to use distilling in encoder, using this argument means not using distilling',
                      default=True)
    args.add_argument('--dropout', type=float, default=0.05, help='dropout')
    args.add_argument('--embed', type=str, default='timeF',
                      help='time features encoding, options:[timeF, fixed, learned]')
    args.add_argument('--activation', type=str, default='gelu', help='activation')
    args.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')

    # forecasting task
    args.add_argument('--seq_len', type=int, default=15, help='input sequence length')
    args.add_argument('--label_len', type=int, default=7, help='start token length')
    args.add_argument('--pred_len', type=int, default=1, help='prediction sequence length')
    args.add_argument('--freq', type=str, default='d',
                      help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')

    # optimization
    # args.add_argument('--itr', type=int, default=2, help='experiments times')
    args.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    args.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    args.add_argument('--patience', type=int, default=3, help='early stopping patience')
    args.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    args.add_argument('--des', type=str, default='test', help='exp description')
    args.add_argument('--loss', type=str, default='mse', help='loss function')
    args.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    args.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    return args.parse_args()


# ------------------------------------------------------------------

if __name__ == "__main__":

    args = get_args()
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    params = Options(args.config_file)
    for run in range(args.multi_run):
        s_path = os.path.join(args.save_path, str(run + 1))
        fix_seed = int(args.seed + (run + 1) * randrange(2022))
        print(f"Seed: {fix_seed}")
        t = Trainer(params, args, s_path, fix_seed)
        t.train()
        t.test()
