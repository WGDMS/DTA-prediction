
CONFIG = {
    "davis": {
        "lr": 1e-3,
        "batch_size": 512,
        "test_batch_size": 512,
        "patience": 50,
        "epochs": {
            "S1": 1000,
            "S2": 200,
            "S3": 500,
            "S4": 200,
        },
    },
    "kiba": {
        "lr": 5e-4,
        "batch_size": 1024,
        "test_batch_size": 1024,
        "patience": 50,
        "epochs": {
            "S1": 1500,
            "S2": 800,
            "S3": 800,
            "S4": 800,
        },
    },
}

SEEDS = [0, 1, 2, 3, 4]

DATASET_MAP = {
    "0": "davis",
    "1": "kiba",
    "davis": "davis",
    "kiba": "kiba",
}


def get_config(dataset_arg, split_arg):
    dataset = DATASET_MAP[str(dataset_arg).lower()]
    split = split_arg.upper()

    cfg = CONFIG[dataset]

    return {
        "dataset": dataset,
        "split": split,
        "lr": cfg["lr"],
        "batch_size": cfg["batch_size"],
        "test_batch_size": cfg["test_batch_size"],
        "patience": cfg["patience"],
        "epochs": cfg["epochs"][split],
        "seeds": SEEDS,
    }


def checkpoint_path(dataset, split, seed):
    return f"models/model_{dataset}_{split}_seed{seed}.pt"


def result_path(dataset, split):
    return f"results/result_{dataset}_{split}_5seeds.txt"