import experiments.keep_pct_readout


def get_default_keep_pct_readout_args():
    # Training settings
    return {
        "batch_size": 64,
        "test_batch_size": 1000,
        "epochs": 5,
        "lr": 0.003,
        "momentum": 0.5,
        "no_cuda": False,
        "log_interval": 100,
        "save_model": False
    }


options = {
   "keep_pct_readout_default":
      lambda train_loader_fn, test_loader_fn, model:
         experiments.keep_pct_readout.run(train_loader_fn, test_loader_fn, model, get_default_keep_pct_readout_args())
}
