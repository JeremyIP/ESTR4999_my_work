import argparse
import importlib
import importlib.util
import os
import re
import subprocess

import lightning.pytorch as L
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import CSVLogger, WandbLogger

from core.data_runner import DataInterface
from core.ltsf_runner import LTSFRunner
from core.util import cal_conf_hash
from core.util import load_module_from_path

from genetic import genetic_algorithm

# Modified Code to invoke call back to print the loss per epoch
from lightning.pytorch.callbacks import Callback
class TrainLossLoggerCallback(Callback):
    def __init__(self):
        super().__init__()
        self.train_losses = []

    def on_train_epoch_end(self, trainer, pl_module):
        """
        Called at the end of the training epoch.
        Collects the average training loss and appends it to the train_losses list.
        """
        # Retrieve the average training loss from callback_metrics
        avg_loss = trainer.callback_metrics.get('train/loss')
        if avg_loss is not None:
            # Append the average loss to the list
            self.train_losses.append(avg_loss.item())
            # Print the average loss for the epoch
            print(f"Epoch {trainer.current_epoch + 1}: Average Train Loss = {avg_loss.item():.4f}")

'''
    def on_train_end(self, trainer, pl_module):
        """
        Called at the end of training.
        Prints the list of average training losses per epoch.
        """
        print("\nTraining Loss per Epoch:")
        for epoch, loss in enumerate(self.train_losses, 1):
            print(f"Epoch {epoch}: {loss:.4f}")
'''

def train_init(hyper_conf, conf):
    if hyper_conf is not None:
        for k, v in hyper_conf.items():
            conf[k] = v
    conf['conf_hash'] = cal_conf_hash(conf, hash_len=10)


    L.seed_everything(conf["seed"])
    save_dir = os.path.join(conf["save_root"], '{}_{}'.format(conf["model_name"], conf["dataset_name"]))
    if "use_wandb" in conf and conf["use_wandb"]:
        run_logger = WandbLogger(save_dir=save_dir, name=conf["conf_hash"], version='seed_{}'.format(conf["seed"]))
    else:
        run_logger = CSVLogger(save_dir=save_dir, name=conf["conf_hash"], version='seed_{}'.format(conf["seed"]))
    conf["exp_dir"] = os.path.join(save_dir, conf["conf_hash"], 'seed_{}'.format(conf["seed"]))

    callbacks = [
        # ModelCheckpoint(
        #     monitor=conf["val_metric"],
        #     mode="min",
        #     save_top_k=1,
        #     save_last=False,
        #     every_n_epochs=1,
        # ),
        # EarlyStopping(
        #     monitor=conf["val_metric"],
        #     mode='min',
        #     patience=conf["es_patience"],
        # ),
        LearningRateMonitor(logging_interval="epoch"),
        TrainLossLoggerCallback(), # Modified Code to invoke call back to print the loss per epoch
    ]


    trainer = L.Trainer(
        devices=conf["devices"],
        precision=conf["precision"] if "precision" in conf else "32-true",
        logger=run_logger,
        callbacks=callbacks,
        max_epochs=conf["max_epochs"],
        gradient_clip_algorithm=conf["gradient_clip_algorithm"] if "gradient_clip_algorithm" in conf else "norm",
        gradient_clip_val=conf["gradient_clip_val"],
        default_root_dir=conf["save_root"],
        limit_val_batches=0, # Disable validation
        check_val_every_n_epoch=0, # No validation every n epoch
    )

    data_module = DataInterface(**conf)
    model = LTSFRunner(**conf)

    return trainer, data_module, model

def train_func(trainer, data_module, model):
    trainer.fit(model=model, datamodule=data_module)
    #trainer.test(model, datamodule=data_module, ckpt_path='best')
    trainer.test(model, datamodule=data_module)

    model.train_plot_losses()
    model.test_plot_losses()

    return trainer, data_module, model



ticker_symbols = ['AAPL']
#, 'MSFT', 'ORCL', 'AMD', 'CSCO', 'ADBE', 'IBM', 'TXN', 'AMAT', 'MU', 'ADI', 'INTC', 'LRCX', 'KLAC', 'MSI', 'GLW', 'HPQ', 'TYL', 'PTC', 'WDC']

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data_root", default="dataset", type=str, help="data root")
    parser.add_argument("-s", "--save_root", default="save", type=str, help="save root")
    parser.add_argument("--devices", default='0,', type=str, help="device' id to use")
    parser.add_argument("--use_wandb", default=0, type=int, help="use wandb")
    parser.add_argument("--seed", type=int, default=1, help="seed")
    parser.add_argument("--model_name", default="DenseRMoK", type=str, help="Model name")
    parser.add_argument("--revin_affine", default=False, type=bool, help="Use revin affine")
    parser.add_argument("--lr", default=0.001, type=float, help="Learning rate")
    parser.add_argument("--batch_size", default=20, type=int, help="Batch size")
    parser.add_argument("--max_epochs", default=20, type=int, help="Maximum number of epochs")
    parser.add_argument("--optimizer", default="AdamW", type=str, help="Optimizer type")
    parser.add_argument("--optimizer_betas", default=(0.95, 0.9), type=eval, help="Optimizer betas")
    parser.add_argument("--optimizer_weight_decay", default=1e-5, type=float, help="Optimizer weight decay")
    parser.add_argument("--lr_scheduler", default='StepLR', type=str, help="Learning rate scheduler")
    parser.add_argument("--lr_step_size", default=5, type=int, help="Learning rate step size")
    parser.add_argument("--lr_gamma", default=0.5, type=float, help="Learning rate gamma")
    parser.add_argument("--gradient_clip_val", default=5, type=float, help="Gradient clipping value")
    parser.add_argument("--val_metric", default="val/loss", type=str, help="Validation metric")
    parser.add_argument("--test_metric", default="test/mae", type=str, help="Test metric")
    parser.add_argument("--es_patience", default=10, type=int, help="Early stopping patience")
    parser.add_argument("--num_workers", default=10, type=int, help="Number of workers for data loading")

    parser.add_argument("--population_size", default=4, type=int, help="Population Size for GA")
    parser.add_argument("--total_generations", default=2, type=int, help="Total number of generations for GA")
    parser.add_argument("--n_features", default=50, type=int, help="Number of features for GA")
    parser.add_argument("--n_hyperparameters", default=11, type=int, help="Number of hyperparameters for GA")

    args = parser.parse_args()
    args.hist_len = 60
    args.pred_len = 1
    args.var_num = 50
    args.freq = 1440 # TO DO ///
    args.data_split = [2000, 0, 500]

    for symbol in ticker_symbols:
        # Before GA
        args.dataset_name = symbol
        args.indicators_bool = [1 for i in range(args.n_features)]
        args.window_size = [1, 1, 1, 1, 1]
        args.WaveKAN = [1]
        args.NaiveFourierKAN = [1]
        args.JacobiKAN = [1]
        args.ChebyKAN = [1]
        args.TaylorKAN = [1]
        args.RBFKAN = [1]

        training_conf = {
            "seed": int(args.seed),
            "data_root": f"dataset/{symbol}",
            "save_root": args.save_root,
            "devices": args.devices,
            "use_wandb": args.use_wandb
        }

        # GA
        indicators_bool, window_size, WaveKAN, NaiveFourierKAN, JacobiKAN, ChebyKAN, TaylorKAN, RBFKAN = genetic_algorithm(training_conf, **vars(args))

        # After GA
        args.indicators_bool = indicators_bool
        args.window_size = window_size
        args.WaveKAN = WaveKAN
        args.NaiveFourierKAN = NaiveFourierKAN
        args.JacobiKAN = JacobiKAN
        args.ChebyKAN = ChebyKAN
        args.TaylorKAN = TaylorKAN
        args.RBFKAN = RBFKAN

        print("\n")
        print(f"For stock {symbol}, optimal model is finally trained below: ")
        trainer, data_module, model = train_init(training_conf, **vars(args))
        trainer, data_module, model = train_func(trainer, data_module, model)
        print("\n")
