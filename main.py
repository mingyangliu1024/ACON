import argparse
import warnings
import sklearn.exceptions


import trainers

warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)


parser = argparse.ArgumentParser()


# ========  Experiments Name ================
parser.add_argument('--save_dir', default='logs', type=str, help='Directory containing all experiments')
parser.add_argument('--experiment_description', default='ACON', type=str, help='Name of your experiment (UCIHAR, HHAR_P, WISDM')
parser.add_argument('--run_description', default='ACON', type=str, help='name of your runs')

# ========= Select the DA methods ============
parser.add_argument('--da_method', default='ACON', type=str)

# ========= Select the DATASET ==============
parser.add_argument('--data_path', default='/data', type=str, help='Path containing dataset')
parser.add_argument('--dataset', default='UCIHAR',type=str)

# ========= Select the BACKBONE ==============
parser.add_argument('--backbone', default='CNN', type=str)

# ========= Experiment settings ===============
parser.add_argument('--num_runs', default=5, type=int, help='Number of consecutive run with different seeds')
parser.add_argument('--device', default='cuda', type=str, help='cpu or cuda')
parser.add_argument('--num_epochs', type=int, default=50)
parser.add_argument('--bs', type=int, default=32, help='batch size')
parser.add_argument('--lr', type=float, default=0.001, help='optimizer learning rate')
parser.add_argument('--weight_decay', type=float, default=1e-4)
parser.add_argument('--start',type=int, default=0)
parser.add_argument('--end', type=int, default=None)
parser.add_argument('-p','--print-freq', type=int, default=10, help='each epoch print num_epochs/p times ')
parser.add_argument('--num_workers', type=int, default=2)
parser.add_argument('--shuffle', action='store_true', help='whether shuffle the train dataset')
parser.add_argument('--phase', default='train', type=str)
parser.add_argument('--test_model_prefix', type=str)

# =========        ACON       ===============
parser.add_argument('--kl_reduction', default='mean')
parser.add_argument('--kl_t',default=1.0, type=float)
parser.add_argument('--disc_hid_dim', type=int, default=128)
# trade_off for different loss
parser.add_argument('--entropy_trade_off', type=float,default=0.01)
parser.add_argument('--domain_trade_off', type=float,default=1.0)
parser.add_argument('--align_s_trade_off', type=float,default=1.0)
parser.add_argument('--align_t_trade_off', type=float,default=1.0)
parser.add_argument('--cls_trade_off', type=float,default=1.0)


args = parser.parse_args()



if __name__ == "__main__":
    
    trainer = trainers.da_trainer(args)
    if args.phase == 'test':
        trainer.test()
    else:
        trainer.train()
    
   