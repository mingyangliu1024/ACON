import argparse
import warnings
import sklearn.exceptions


import trainers



warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)


parser = argparse.ArgumentParser()


# ========  Experiments Name ================
parser.add_argument('--save_dir',               default='logs',         type=str, help='Directory containing all experiments')
parser.add_argument('--experiment_description', default='ACON',               type=str, help='Name of your experiment (HAR, HHAR_P, WISDM')
parser.add_argument('--run_description',        default='ACON',                     type=str, help='name of your runs')

# ========= Select the DA methods ============
parser.add_argument('--da_method',              default='ACON',               type=str)

# ========= Select the DATASET ==============
parser.add_argument('--data_path',              default=r'/data/',                  type=str, help='Path containing dataset')
parser.add_argument('--dataset',                default='HAR',                      type=str)

# ========= Select the BACKBONE ==============
parser.add_argument('--backbone',               default='CNN',                      type=str, help='Backbone of choice: (CNN - RESNET18 - TCN)')

# ========= Experiment settings ===============
parser.add_argument('--num_runs',               default=1,                          type=int, help='Number of consecutive run with different seeds')
parser.add_argument('--device',                 default='cuda',                   type=str, help='cpu or cuda')





# parameter of datasets


# Hyperparameters for timensnet
parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')

parser.add_argument('--top_k', type=int, default=3, help='for TimesBlock')
parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
parser.add_argument('--d_model', type=int, default=64, help='dimension of model')
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
parser.add_argument('--d_ff', type=int, default=64, help='dimension of fcn')
parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')


# Hyperparameters for Dlinear
parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
parser.add_argument('--no-share', action='store_true', help='whether shared model among different variates')
parser.add_argument('--dropout', type=float, default=0.1, help='dropout')    




# Hyperparameters for denoising_da
parser.add_argument('--mode',type=int,default=32)
parser.add_argument('--extract-trend', action='store_true', help='whether extract trend before extract temporal feature')
parser.add_argument('--feature-domains', type=str, default='tf', choices=['t', 'f', 'tf'])
parser.add_argument('--align-loss', type=str, default='SinkhornDistance', choices=['SinkhornDistance', 'd_svd_l1', 'd_svd_l2'])
parser.add_argument('--topk',type=float, default=0.8, help='align the top batch_size*topk value of s')
parser.add_argument('--svd-trade-off',type=float, default=0.1)



# Training
parser.add_argument('--num_epochs', type=int, default=50)
parser.add_argument('--bs', type=int, default=32)
parser.add_argument('--lr', type=float, default=0.001, help='optimizer learning rate')
parser.add_argument('--weight_decay', type=float, default=1e-4)
parser.add_argument('--select_type', type=str, default='t', choices=['f', 't'])
parser.add_argument('--note', type=str, default=' ')
parser.add_argument('--start',type=int, default=0)
parser.add_argument('--end', type=int, default=None)

# Frequancy domain
parser.add_argument('--no-cos', action='store_true', help='whether smooth the input before FFT')
parser.add_argument('--topk-period',type=int,default=1, help='the topk largest periods are choosen')
parser.add_argument('--period_list', default=None,nargs='+', type=int)
parser.add_argument('--concat_between_periods', action='store_true')
parser.add_argument('--channel_weighted_mean_between_periods', action='store_true')
parser.add_argument('--concat_in_periods', action='store_true')
parser.add_argument('--mean_period_in_datasets', action='store_true')
parser.add_argument('--fft_normalize',action='store_true', help='whether normalize the spectral data')
parser.add_argument('--fft_a_normalize',action='store_true', help='whether normalize the amplitude')
parser.add_argument('--fft_mode', default=100000, type=int)
parser.add_argument('--frequency2temporal', action='store_true')
parser.add_argument('--frequency2temporal_detach', action='store_true')
parser.add_argument('--source_target_respectively', action='store_true')
parser.add_argument('--source_f2t',action='store_true')
parser.add_argument('--target_t2f',action='store_true')
parser.add_argument('--kl_reduction', default='mean')
parser.add_argument('--kl_t',default=1.0, type=float)

# data and logging
parser.add_argument('-p','--print-freq', type=int, default=10, help='each epoch print num_epochs/p times ')
parser.add_argument('--num_workers', type=int, default=2)
parser.add_argument('--shuffle', action='store_true', help='whether shuffle the train dataset')

# f_as_y
parser.add_argument('--cdan_mode',type=int, default=32)
parser.add_argument('--disc_hid_dim', type=int, default=512)

# choice for trainer
parser.add_argument('--trainer', type=str, choices=['da','raincoat', 'data', 'MDD'], default='raincoat')

# trade_off for different loss
parser.add_argument('--cent_trade_off', type=float,default=0.01)
parser.add_argument('--domain_trade_off', type=float,default=0.5)
parser.add_argument('--align_s_trade_off', type=float,default=1.0)
parser.add_argument('--align_t_trade_off', type=float,default=1.0)
parser.add_argument('--cls_trade_off', type=float,default=0.01)



# adversarial
parser.add_argument('--alpha',type=float, default=1.0)
parser.add_argument('--warm_up',action='store_true', )
parser.add_argument('--grl_coeff', default=1.0, type=float)
parser.add_argument('--disc_layer',type=int,default=3)



# cluda hyper
parser.add_argument('--weight_loss_src', type=float, default=1.0)
parser.add_argument('--weight_loss_trg', type=float, default=1.0)
parser.add_argument('--weight_loss_ts', type=float, default=1.0)
parser.add_argument('--hidden_dim_MLP', type=int, default=128)


# mdd hyper
parser.add_argument('--mdd_trade_off', type=float, default=1.0, help='the trade-off hyperparameter for transfer loss')
parser.add_argument('--margin', type=float, default=4., help="margin gamma")

# others
parser.add_argument('--focal_loss', action='store_true')
parser.add_argument('--phase', default='train', type=str)
parser.add_argument('--src_id', type=str, default='0')
parser.add_argument('--trg_id', type=str, default='1')
parser.add_argument('--model_path', type=str)
parser.add_argument('--run_id', type=int, default=0)


args = parser.parse_args()





if __name__ == "__main__":
    if args.trainer == 'raincoat':
        trainer = trainers.cross_domain_trainer(args)
    elif args.trainer == 'da':
        trainer = trainers.da_trainer(args)

   
    
    if args.phase == 'test':
        trainer.test()
    else:
        trainer.train()
    
   