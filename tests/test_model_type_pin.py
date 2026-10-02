"""learning.training.model_type (live 2026-09-30, Rei's model-validation fix): pin the model family."""

from typesafe_mario.adapt1_policy import FULL_V2_FEATURE_NAMES, build_domain_config
from typesafe_mario.offline_client import OfflineAdapt1Client


def test_pinned_family_lands_in_learning_training():
    cfg = build_domain_config(
        "d", feature_names=FULL_V2_FEATURE_NAMES, sequential=False, model_type="extra_trees"
    )
    assert cfg["learning"]["training"] == {"enabled": True, "model_type": "extra_trees"}


def test_default_leaves_the_server_on_auto():
    cfg = build_domain_config("d", feature_names=FULL_V2_FEATURE_NAMES, sequential=False)
    assert "training" not in cfg["learning"]


def test_offline_contract_accepts_the_block():
    client = OfflineAdapt1Client()
    status, _ = client.create_domain(
        build_domain_config(
            "d", feature_names=FULL_V2_FEATURE_NAMES, sequential=False, model_type="extra_trees"
        )
    )
    assert status == 201
