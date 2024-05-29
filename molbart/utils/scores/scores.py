from typing import Any, Dict, List, Optional

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs

from molbart.utils import smiles_utils


class BaseScore:
    """
    Base scoring class.
    """

    scorer_name = "base"

    def __init__(self, **kwargs: Any):
        return

    def __call__(self, sampled_smiles: List[List[str]], target_smiles: Optional[List[str]] = None) -> Dict[str, float]:
        return self._score_sampled_smiles(sampled_smiles, target_smiles)

    def __repr__(self):
        repr_name = self.scorer_name
        return repr_name

    def _score_sampled_smiles(
        self, sampled_smiles: List[List[str]], target_smiles: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Scoring function which should be implemented in each new Score class."""
        raise NotImplementedError("self._score_sampled_smiles() needs to be implemented for every scoring class.")


class FractionInvalidScore(BaseScore):
    """
    Scoring using fraction of invalid of all or top-1 SMILES.
    """

    scorer_name = "fraction_invalid"

    def __init__(self, only_top1: bool = False):
        """
        Args:
            only_top1: If True, will only compute fraction of invalid top-1 SMILES,
                otherwise fraction invalid is over all generated SMILES.
        """
        super().__init__()
        self.only_top1 = only_top1

    def _score_sampled_smiles(
        self, sampled_smiles: List[List[str]], target_smiles: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """Computing fraction of invalid SMILES."""

        if self.only_top1:
            is_valid = [
                bool(Chem.MolFromSmiles(top_k_smiles[0])) if len(top_k_smiles) > 0 else False
                for top_k_smiles in sampled_smiles
            ]
        else:
            is_valid = []
            for top_k_smiles in sampled_smiles:
                for smiles in top_k_smiles:
                    is_valid.append(bool(Chem.MolFromSmiles(smiles)))

        fraction_invalid = 1 - (sum(is_valid) / len(is_valid))
        return {self.scorer_name: fraction_invalid}


class FractionUniqueScore(BaseScore):
    """
    Scoring using the fraction of uniquely sampled SMILES among the top-N sampled SMILES.
    """

    scorer_name = "fraction_unique"

    def __init__(self, canonicalized: bool = False, only_valid: bool=True):
        """
        Args:
            canonicalized: whether the sampled_smiles and target_smiles are
                been canonicalized.
            only_valid: whether to only consider valid SMILES yielding molecules.
        """
        super().__init__()
        self._canonicalized = canonicalized
        self._only_valid = only_valid

    def _score_sampled_smiles(
        self, sampled_smiles: List[List[str]], target_smiles: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """Computing fraction of unique top-N SMILES."""

        n_samples = len(sampled_smiles)
        n_beams = len(sampled_smiles[0])

        n_unique_total = 0
        for top_k in sampled_smiles:
            if not self._canonicalized:
                if self._only_valid:
                    top_k = [smiles_utils.inchi_key(smiles) for smiles in top_k if Chem.MolFromSmiles(smiles)]
                else:
                    top_k = [smiles_utils.inchi_key(smiles) for smiles in top_k]
            elif self._only_valid:
                top_k = [smiles for smiles in top_k if Chem.MolFromSmiles(smiles)]
            n_unique = len(set(top_k))
            n_unique_total += n_unique
        fraction_unique = n_unique_total / (n_beams * n_samples)
        return {self.scorer_name: fraction_unique}


class TanimotoSimilarityScore(BaseScore):
    """
    Scoring using the Tanomoto similarity of the top-1 sampled SMILES and the target
    SMILES.
    """

    scorer_name = "top1_tanimoto_similarity"

    def __init__(self, statistics="mean"):
        """
        Args:
            return_strategy: ["mean", "median", "all"], returns the average similarity or
            all similarities.
        """
        super().__init__()

        if statistics not in ["mean", "median", "all"]:
            raise ValueError(f"'statistics' should be either 'mean', 'median' or 'all'," f" not {statistics}")
        self._statistics = statistics

        self._stat_fcn = {"mean": np.mean, "median": np.median}

    def _get_statistics(self, similarities: np.ndarray) -> float:
        if self._statistics == "all":
            return [similarities]
        similarities = similarities[~np.isnan(similarities)]
        return self._stat_fcn[self._statistics](similarities)

    def _score_sampled_smiles(
        self, sampled_smiles: List[List[str]], target_smiles: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Compute similarities of ECPF4 fingerprints of target and top-1 sampled molecules.
        """
        target_molecules = [Chem.MolFromSmiles(smiles) for smiles in target_smiles]

        sampled_molecules = [
            Chem.MolFromSmiles(smiles_list[0]) if len(smiles_list) > 0 else None for smiles_list in sampled_smiles
        ]

        n_samples = len(target_molecules)

        similarities = np.nan * np.ones(n_samples)
        counter = 0
        for sampled_mol, target_mol in zip(sampled_molecules, target_molecules):
            if not sampled_mol or not target_mol:
                counter += 1
                continue

            fp1 = AllChem.GetMorganFingerprint(sampled_mol, 2)
            fp2 = AllChem.GetMorganFingerprint(target_mol, 2)

            similarities[counter] = DataStructs.TanimotoSimilarity(fp1, fp2)  # Tanimoto similarity = Jaccard similarity
            counter += 1

        return {self.scorer_name: self._get_statistics(similarities)}


class TopKAccuracyScore(BaseScore):
    scorer_name = "top_k_accuracy"

    def __init__(
        self,
        top_ks: np.ndarray = np.array([1, 3, 5, 10, 20, 30, 40, 50]),
        canonicalized: bool = False,
    ):
        """
        Args:
            top_ks: a list of top-Ks to compute accuracy for.
            canonicalized: whether the sampled_smiles and target_smiles are
                been canonicalized.
        """
        super().__init__()
        self._top_ks = top_ks
        self._canonicalized = canonicalized

    def _is_in_set(self, sampled_smiles: List[List[str]], target_smiles: List[str], k: int) -> np.ndarray:
        if not self._canonicalized:
            target_smiles = [smiles_utils.canonicalize_smiles(smiles) for smiles in target_smiles]

            sampled_smiles = [
                [smiles_utils.canonicalize_smiles(smiles) for smiles in smiles_list] for smiles_list in sampled_smiles
            ]

        is_in_set = [
            tgt_smi in sampled_smi[0:k] if len(sampled_smi[0:k]) > 0 else False
            for sampled_smi, tgt_smi in zip(sampled_smiles, target_smiles)
        ]
        return is_in_set

    def _score_sampled_smiles(self, sampled_smiles: List[List[str]], target_smiles: List[str]) -> Dict[str, float]:
        n_beams = np.max(np.array([1, np.max(np.asarray([len(smiles) for smiles in sampled_smiles]))]))
        top_ks = self._top_ks[self._top_ks <= n_beams]

        columns = []
        is_in_set = np.zeros((len(sampled_smiles), len(top_ks)), dtype=bool)
        for i_k, k in enumerate(top_ks):
            columns.append(f"accuracy_top_{k}")
            is_in_set[:, i_k] = self._is_in_set(sampled_smiles, target_smiles, k)

        is_in_set = np.cumsum(is_in_set, axis=1)
        top_n_accuracy = np.mean(is_in_set > 0, axis=0)

        if max(top_ks) == 1:
            return {"accuracy": top_n_accuracy[0]}

        scores = {col: accuracy for col, accuracy in zip(columns, top_n_accuracy)}
        return scores
