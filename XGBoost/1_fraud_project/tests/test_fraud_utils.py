from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))


class FraudUtilsTest(unittest.TestCase):
    def test_merge_transaction_identity_keeps_target_and_submission_index(self) -> None:
        from fraud_utils.data import merge_transaction_identity

        transaction = pd.DataFrame(
            {
                "TransactionID": [1, 2],
                "isFraud": [0, 1],
                "TransactionAmt": [10.0, 20.0],
            }
        ).set_index("TransactionID")
        identity = pd.DataFrame({"TransactionID": [2], "id_01": [5.0]}).set_index(
            "TransactionID"
        )

        merged = merge_transaction_identity(transaction, identity)

        self.assertEqual(list(merged.index), [1, 2])
        self.assertIn("isFraud", merged.columns)
        self.assertIn("id_01", merged.columns)
        self.assertTrue(pd.isna(merged.loc[1, "id_01"]))

    def test_prepare_features_encodes_objects_and_removes_training_only_columns(
        self,
    ) -> None:
        from fraud_utils.features import prepare_features

        train = _minimal_ieee_frame(
            transaction_ids=[1, 2, 3, 4],
            fraud=[0, 1, 0, 1],
            product=["W", "C", "W", "H"],
        )
        test = _minimal_ieee_frame(
            transaction_ids=[5, 6],
            fraud=None,
            product=["W", "S"],
        )

        prepared = prepare_features(train, test, drop_useless=False, fill_value=-999)

        self.assertEqual(prepared.y.tolist(), [0, 1, 0, 1])
        self.assertNotIn("isFraud", prepared.x_train.columns)
        self.assertNotIn("DT", prepared.x_train.columns)
        self.assertNotIn("TransactionDT", prepared.x_train.columns)
        self.assertEqual(int(prepared.x_train.isna().sum().sum()), 0)
        self.assertEqual(int(prepared.x_test.isna().sum().sum()), 0)
        self.assertEqual(
            prepared.x_train.select_dtypes(include=["object"]).columns.tolist(),
            [],
        )
        self.assertTrue(
            {"uid", "uid2", "uid3", "TransactionAmt"}.issubset(
                prepared.x_train.columns
            )
        )

    def test_feature_groups_capture_common_ieee_families(self) -> None:
        from fraud_utils.analysis import assign_feature_groups

        groups = assign_feature_groups(
            [
                "TransactionAmt",
                "card1",
                "addr1",
                "C1",
                "D15",
                "V12",
                "id_31",
                "P_emaildomain",
                "DeviceInfo",
                "uid_TransactionAmt_mean",
                "DT_hour",
                "unknown_col",
            ]
        )

        self.assertEqual(groups["TransactionAmt"], "amount")
        self.assertEqual(groups["card1"], "card")
        self.assertEqual(groups["addr1"], "addr")
        self.assertEqual(groups["C1"], "C_count")
        self.assertEqual(groups["D15"], "D_time_delta")
        self.assertEqual(groups["V12"], "V_anonymous")
        self.assertEqual(groups["id_31"], "identity")
        self.assertEqual(groups["P_emaildomain"], "email")
        self.assertEqual(groups["DeviceInfo"], "device")
        self.assertEqual(groups["uid_TransactionAmt_mean"], "uid_aggregate")
        self.assertEqual(groups["DT_hour"], "time")
        self.assertEqual(groups["unknown_col"], "other")

    def test_synergy_score_identifies_joint_extra_drop(self) -> None:
        from fraud_utils.experiments import compute_permutation_synergy

        result = compute_permutation_synergy(
            baseline_score=0.90,
            single_scores={"a": 0.86, "b": 0.87},
            joint_scores={("a", "b"): 0.80},
        )

        self.assertEqual(result.loc[0, "feature_a"], "a")
        self.assertEqual(result.loc[0, "feature_b"], "b")
        self.assertTrue(np.isclose(result.loc[0, "drop_a"], 0.04))
        self.assertTrue(np.isclose(result.loc[0, "drop_b"], 0.03))
        self.assertTrue(np.isclose(result.loc[0, "joint_drop"], 0.10))
        self.assertTrue(np.isclose(result.loc[0, "synergy"], 0.03))

    def test_select_top_features_preserves_requested_columns(self) -> None:
        from fraud_utils.experiments import select_top_features

        importance = pd.DataFrame(
            {
                "feature": ["a", "b", "c"],
                "importance": [0.1, 0.9, 0.4],
            }
        )

        selected = select_top_features(
            importance,
            top_k=2,
            always_keep=["TransactionAmt", "a"],
        )

        self.assertEqual(selected, ["TransactionAmt", "a", "b", "c"])


def _minimal_ieee_frame(
    transaction_ids: list[int],
    fraud: list[int] | None,
    product: list[str],
) -> pd.DataFrame:
    frame = pd.DataFrame(index=pd.Index(transaction_ids, name="TransactionID"))
    if fraud is not None:
        frame["isFraud"] = fraud
    frame["TransactionDT"] = [86_400 * idx for idx in range(len(transaction_ids))]
    frame["TransactionAmt"] = [10.0 + idx for idx in range(len(transaction_ids))]
    frame["ProductCD"] = product
    frame["card1"] = [1000 + idx % 2 for idx in range(len(transaction_ids))]
    frame["card2"] = [200.0 + idx % 2 for idx in range(len(transaction_ids))]
    frame["card3"] = [150.0 for _ in transaction_ids]
    frame["card5"] = [226.0 for _ in transaction_ids]
    frame["addr1"] = [100.0 + idx % 2 for idx in range(len(transaction_ids))]
    frame["addr2"] = [87.0 for _ in transaction_ids]
    frame["D9"] = [np.nan, 0.5] * (len(transaction_ids) // 2) + [np.nan] * (
        len(transaction_ids) % 2
    )
    frame["P_emaildomain"] = ["gmail.com", None] * (len(transaction_ids) // 2) + [
        "yahoo.com"
    ] * (len(transaction_ids) % 2)
    frame["R_emaildomain"] = ["gmail.com", "hotmail.com"] * (
        len(transaction_ids) // 2
    ) + [None] * (len(transaction_ids) % 2)
    frame["id_31"] = ["chrome 69.0", None] * (len(transaction_ids) // 2) + [
        "firefox 60.0"
    ] * (len(transaction_ids) % 2)
    frame["DeviceInfo"] = ["SM-G950", None] * (len(transaction_ids) // 2) + [
        "Moto G"
    ] * (len(transaction_ids) % 2)
    frame["C1"] = [1.0 for _ in transaction_ids]
    frame["D1"] = [1.0 + idx for idx in range(len(transaction_ids))]
    frame["dist1"] = [np.nan for _ in transaction_ids]
    return frame


if __name__ == "__main__":
    unittest.main()
