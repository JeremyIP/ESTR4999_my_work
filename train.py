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

    def on_train_end(self, trainer, pl_module):
        """
        Called at the end of training.
        Prints the list of average training losses per epoch.
        """
        print("\nTraining Loss per Epoch:")
        for epoch, loss in enumerate(self.train_losses, 1):
            print(f"Epoch {epoch}: {loss:.4f}")


'''
def load_config(exp_conf_path):
    # 加载 exp_conf
    exp_conf = load_module_from_path("exp_conf", exp_conf_path).exp_conf

    # 加载 task_conf
    task_conf_module = importlib.import_module('config.base_conf.task')
    task_conf = task_conf_module.task_conf

    # 加载 data_conf
    data_conf_module = importlib.import_module('config.base_conf.datasets')
    data_conf = eval('data_conf_module.{}_conf'.format(exp_conf['dataset_name']))

    # conf 融合，参数优先级: exp_conf > task_conf = data_conf
    fused_conf = {**task_conf, **data_conf}
    fused_conf.update(exp_conf)

    return fused_conf
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

    model.plot_losses()



ticker_symbols = ['AAPL']
#, 'MSFT', 'ORCL', 'AMD', 'CSCO', 'ADBE', 'IBM', 'TXN', 'AMAT', 'MU', 'ADI', 'INTC', 'LRCX', 'KLAC', 'MSI', 'GLW', 'HPQ', 'TYL', 'PTC', 'WDC']

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("-d", "--data_root", default="dataset", type=str, help="data root")
    parser.add_argument("-s", "--save_root", default="save", type=str, help="save root")
    parser.add_argument("--devices", default='0,', type=str, help="device' id to use")
    parser.add_argument("--use_wandb", default=0, type=int, help="use wandb")
    parser.add_argument("--seed", type=int, default=1, help="seed")
    
    # Adding new arguments
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

    args = parser.parse_args()

    # Now, args will contain all the attributes set via command line or defaults

    # parser.add_argument("-c", "--config", type=str)

    for symbol in ticker_symbols:

        # Before GA
        args.dataset_name = symbol

        '''
         # Using regular expressions to match the pattern _{int}for{int}.py
        pattern = re.compile(rf"{symbol}_(\d+)for(\d+)\.py")
        matching_files = [config_file for config_file in os.listdir("config/reproduce_conf/RMoK/") if pattern.match(config_file)]
        
        if matching_files:
            args.config = f"config/reproduce_conf/RMoK/{matching_files[0]}"
        else:
            print(f"No matching config file found for {symbol}.")
        '''

        args.hist_len = 60
        args.pred_len = 1
        args.var_num = 50
        args.freq = 1440 # TO DO ///
        args.data_split = [2000, 0, 500]
        
        

        # init_exp_conf = load_config(args.config)
        

        training_conf = {
            "data_root": f"dataset/{symbol}",
            "save_root": args.save_root,
            "devices": args.devices,
            "use_wandb": args.use_wandb,
            "seed": int(args.seed)
        }


        ''' # TO DO ///
        trainer, data_module, model = train_init(training_conf, init_exp_conf)

        # Run the genetic algorithm
        population_size = 10
        total_generations = 10
        n_features = 50
        n_hyperparameters = 11

        best_solution = genetic_algorithm(population_size, total_generations, n_features, n_hyperparameters, trainer, data_module, model)
        print("Best Solution found:", best_solution)

        training_conf["features_mask"] = best_solution.genes["features"]
        print(training_conf["features_mask"])
        trainer, data_module, model = train_init(training_conf, init_exp_conf)
        train_func(trainer, data_module, model)
        '''
        trainer, data_module, model = train_init(training_conf, args) 
        train_func(trainer, data_module, model)
        #trainer, data_module, model = train_init(training_conf, init_exp_conf) # Train final optimal model
        
        
