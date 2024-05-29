import hydra

import molbart.utils.data_utils as util
from molbart.models import Chemformer


@hydra.main(version_base=None, config_path="config", config_name="inference_score")
def main(args):
    util.seed_everything(args.seed)

    print("Running model inference and scoring.")

    chemformer = Chemformer(args)

    chemformer.score_model(
        n_unique_beams=args.n_unique_beams,
        dataset=args.dataset_part,
        output_scores=args.output_score_data,
        output_sampled_smiles=args.output_sampled_smiles,
    )
    print("Model inference and scoring done.")
    return


if __name__ == "__main__":
    main()
