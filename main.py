comet_support = True
try:
    from comet_ml import Experiment
except ImportError as e:
    print("Comet ML is not installed, ignore the comet experiment monitor")
    comet_support = False
from models import DRPNet
from time import time
from utils import set_seed, graph_collate_func, mkdir
from configs import get_cfg_defaults
from dataloader import DTIDataset, MultiDataLoader
from torch.utils.data import DataLoader
from trainer import Trainer
import torch
import argparse
import warnings, os
import pandas as pd

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser(description="DRPNet for drug prediction")
parser.add_argument('--cfg', required=True, help="path to config file", type=str)
parser.add_argument('--data', required=True, type=str, metavar='TASK',
                    help='dataset')
args = parser.parse_args()


def main():
    torch.cuda.empty_cache()
    warnings.filterwarnings("ignore", message="invalid value encountered in divide")
    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.cfg)
    set_seed(cfg.SOLVER.SEED)
    suffix = str(int(time() * 1000))[6:]
    mkdir(cfg.RESULT.OUTPUT_DIR)
    experiment = None
    print(f"Config yaml: {args.cfg}")
    print(f"Hyperparameters: {dict(cfg)}")
    print(f"Running on: {device}", end="\n\n")

    dataFolder = f'./datasets/{args.data}'

    train_path = os.path.join(dataFolder, 'train.csv')
    val_path = os.path.join(dataFolder, "test.csv")
    test_path = os.path.join(dataFolder, "test.csv")
    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)
    df_test = pd.read_csv(test_path)
    # 根据路径推断数据类型
    if "train.csv" in train_path:
        data_type = "train"
        print(f"Inferred data type for train path: {data_type}")
    elif "test.csv" in val_path:
        data_type = "val"
        print(f"Inferred data type for val path: {data_type}")
    elif "test.csv" in test_path:
        data_type = "test"
        print(f"Inferred data type for test path: {data_type}")
    print(f"Train path: {train_path}")
    print(f"Val path: {val_path}")
    print(f"Test path: {test_path}")

    train_dataset = DTIDataset(df_train.index.values, df_train, data_type="train")
    val_dataset = DTIDataset(df_val.index.values, df_val, data_type="val")
    test_dataset = DTIDataset(df_test.index.values, df_test, data_type="test")

    if cfg.COMET.USE and comet_support:
        experiment = Experiment(
            project_name=cfg.COMET.PROJECT_NAME,
            workspace=cfg.COMET.WORKSPACE,
            auto_output_logging="simple",
            log_graph=True,
            log_code=False,
            log_git_metadata=False,
            log_git_patch=False,
            auto_param_logging=False,
            auto_metric_logging=False
        )
        hyper_params = {
            "LR": cfg.SOLVER.LR,
            "Output_dir": cfg.RESULT.OUTPUT_DIR,
        }
        experiment.log_parameters(hyper_params)
        if cfg.COMET.TAG is not None:
            experiment.add_tag(cfg.COMET.TAG)
        experiment.set_name(f"{args.data}_{suffix}")

    params = {'batch_size': cfg.SOLVER.BATCH_SIZE, 'shuffle': True, 'num_workers': cfg.SOLVER.NUM_WORKERS,
              'drop_last': True, 'collate_fn': graph_collate_func}
    print(f"Batch size: {cfg.SOLVER.BATCH_SIZE}")
    print(f"Drop last: {params['drop_last']}")

    training_generator = DataLoader(train_dataset, **params)
    params['shuffle'] = False
    params['drop_last'] = False
    val_generator = DataLoader(val_dataset, **params)
    test_generator = DataLoader(test_dataset, **params)
    print("Training generator first element types:")
    first_element = next(iter(training_generator))
    print(f"Type of v_d: {type(first_element[0])}")
    print(f"Type of smiles_string: {type(first_element[1])}")
    print(f"Type of drug_smiles_string: {type(first_element[2])}")  # 修改这里
    print(f"Type of y: {type(first_element[3])}")
    print(f"Type of feature_tensor: {type(first_element[4])}")

    model = DRPNet(**cfg).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=cfg.SOLVER.LR)
    print(f"Learning Rate from config: {cfg.SOLVER.LR}")
    torch.backends.cudnn.benchmark = True

    trainer = Trainer(model, opt, device, training_generator, val_generator, test_generator, experiment=experiment,
                      **cfg)
    result = trainer.train()

    with open(os.path.join(cfg.RESULT.OUTPUT_DIR, "model_architecture.txt"), "w") as wf:
        wf.write(str(model))

    print()
    print(f"Directory for saving result: {cfg.RESULT.OUTPUT_DIR}")

    return result


if __name__ == '__main__':
    s = time()
    result = main()
    e = time()
    print(f"Total running time: {round(e - s, 2)}s")