import time

import hydra

import molbart.utils.data_utils as util
from molbart.models import Chemformer


@hydra.main(version_base=None, config_path="config", config_name="fine_tune")
def main(args):
    util.seed_everything(args.seed)
    print("Fine-tuning CHEMFORMER.")
    chemformer = Chemformer(args)
    t0 = time.time()
    chemformer.fit()
    t_fit = time.time() - t0
    print(f"Training complete, time: {t_fit}")
    print("Done fine-tuning.")
    return


if __name__ == "__main__":
    main()
