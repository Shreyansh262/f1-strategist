"""
tests/test_transformer.py

Unit tests for the TabTransformer model.
Tests forward pass, output shape, era conditioning, MC dropout,
and uncertainty estimation — without requiring trained weights.

Run with:
    pytest tests/test_transformer.py -v
"""

import numpy as np
import pytest
import torch

from src.models.lap_time.train_transformer import (
    CAT_CARDINALITIES,
    CONT_FEATURE_INDICES,
    EMB_DIM,
    FFN_DIM,
    MC_SAMPLES,
    N_HEADS,
    N_TRANSFORMER,
    TabTransformer,
    TransformerBlock,
    predict_with_uncertainty,
)
from src.pipeline.features import (
    CAT_FEATURE_INDICES,
    MODEL_FEATURE_COLUMNS,
)

BATCH = 16
N_CAT  = len(CAT_FEATURE_INDICES)
N_CONT = len(CONT_FEATURE_INDICES)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def model() -> TabTransformer:
    return TabTransformer(
        cat_cardinalities=CAT_CARDINALITIES,
        n_cont=N_CONT,
    )


def make_batch(batch_size: int = BATCH):
    """Create synthetic input tensors."""
    x_cat  = torch.randint(0, 5, (batch_size, N_CAT))
    x_cont = torch.randn(batch_size, N_CONT)
    return x_cat, x_cont


# ---------------------------------------------------------------------------
# TransformerBlock
# ---------------------------------------------------------------------------

class TestTransformerBlock:

    def test_output_shape_unchanged(self):
        block = TransformerBlock(dim=32, n_heads=4, ffn_dim=64, dropout=0.0)
        x = torch.randn(BATCH, N_CAT, 32)
        out = block(x)
        assert out.shape == x.shape

    def test_residual_connection_exists(self):
        """Output should not be all zeros even with zero input due to LayerNorm bias."""
        block = TransformerBlock(dim=32, n_heads=4, ffn_dim=64, dropout=0.0)
        x = torch.zeros(BATCH, N_CAT, 32)
        out = block(x)
        assert not torch.allclose(out, torch.zeros_like(out))


# ---------------------------------------------------------------------------
# TabTransformer forward pass
# ---------------------------------------------------------------------------

class TestTabTransformerForward:

    def test_output_shape(self, model):
        x_cat, x_cont = make_batch()
        out = model(x_cat, x_cont)
        assert out.shape == (BATCH, 1)

    def test_output_is_finite(self, model):
        x_cat, x_cont = make_batch()
        out = model(x_cat, x_cont)
        assert torch.isfinite(out).all()

    def test_output_is_float(self, model):
        x_cat, x_cont = make_batch()
        out = model(x_cat, x_cont)
        assert out.dtype == torch.float32

    def test_single_sample(self, model):
        """Model must handle batch size of 1."""
        x_cat, x_cont = make_batch(batch_size=1)
        out = model(x_cat, x_cont)
        assert out.shape == (1, 1)

    def test_large_batch(self, model):
        x_cat, x_cont = make_batch(batch_size=512)
        out = model(x_cat, x_cont)
        assert out.shape == (512, 1)

    def test_sentinel_minus_one_handled(self, model):
        """CircuitEncoded=-1 (unseen circuit) must not crash — clamped to 0."""
        x_cat, x_cont = make_batch()
        x_cat[:, 0] = -1   # CircuitEncoded = -1 for all rows
        out = model(x_cat, x_cont)
        assert torch.isfinite(out).all()

    def test_does_not_mutate_inputs(self, model):
        x_cat, x_cont = make_batch()
        x_cat_copy  = x_cat.clone()
        x_cont_copy = x_cont.clone()
        _ = model(x_cat, x_cont)
        assert torch.equal(x_cat,  x_cat_copy)
        assert torch.equal(x_cont, x_cont_copy)


# ---------------------------------------------------------------------------
# Era conditioning
# ---------------------------------------------------------------------------

class TestEraConditioning:

    def test_era_0_and_era_1_produce_different_outputs(self, model):
        """
        Era=0 and Era=1 should produce different predictions for identical
        tyre/fuel/weather inputs. If they don't, the era embedding isn't working.
        """
        x_cat, x_cont = make_batch()
        x_cat_era0 = x_cat.clone()
        x_cat_era1 = x_cat.clone()
        x_cat_era0[:, 2] = 0   # Era column index = 2
        x_cat_era1[:, 2] = 1

        model.eval()
        with torch.no_grad():
            out_era0 = model(x_cat_era0, x_cont)
            out_era1 = model(x_cat_era1, x_cont)

        assert not torch.allclose(out_era0, out_era1), \
            "Era=0 and Era=1 produce identical outputs — era embedding not working"

    def test_era_embedding_dimension(self, model):
        """Era embedding table should have shape [n_era+1, emb_dim]."""
        # Era is the 3rd embedding (index 2)
        era_emb = model.embeddings[2]
        assert era_emb.weight.shape[1] == EMB_DIM


# ---------------------------------------------------------------------------
# Circuit conditioning
# ---------------------------------------------------------------------------

class TestCircuitConditioning:

    def test_different_circuits_produce_different_outputs(self, model):
        x_cat, x_cont = make_batch()
        x_cat_monza  = x_cat.clone()
        x_cat_monaco = x_cat.clone()
        x_cat_monza[:, 0]  = 10
        x_cat_monaco[:, 0] = 5

        model.eval()
        with torch.no_grad():
            out_monza  = model(x_cat_monza,  x_cont)
            out_monaco = model(x_cat_monaco, x_cont)

        assert not torch.allclose(out_monza, out_monaco), \
            "Different circuits produce identical outputs — circuit embedding not working"


# ---------------------------------------------------------------------------
# MC Dropout uncertainty
# ---------------------------------------------------------------------------

class TestMCDropout:

    def test_uncertainty_shape(self, model):
        x_cat, x_cont = make_batch()
        mean_p, std_p = predict_with_uncertainty(model, x_cat, x_cont, n_samples=10)
        assert mean_p.shape == (BATCH,)
        assert std_p.shape  == (BATCH,)

    def test_uncertainty_is_positive(self, model):
        x_cat, x_cont = make_batch()
        _, std_p = predict_with_uncertainty(model, x_cat, x_cont, n_samples=10)
        assert (std_p >= 0).all()

    def test_mc_samples_produce_variance(self, model):
        """With dropout=0.1, repeated forward passes should differ."""
        x_cat, x_cont = make_batch()
        _, std_p = predict_with_uncertainty(model, x_cat, x_cont, n_samples=30)
        # At least some predictions should have non-zero uncertainty
        assert std_p.mean() > 0, "MC dropout produced zero variance — dropout may be disabled"

    def test_eval_mode_is_deterministic(self, model):
        """model.eval() should give deterministic outputs (no dropout)."""
        x_cat, x_cont = make_batch()
        model.eval()
        with torch.no_grad():
            out1 = model(x_cat, x_cont)
            out2 = model(x_cat, x_cont)
        assert torch.allclose(out1, out2)


# ---------------------------------------------------------------------------
# Parameter count
# ---------------------------------------------------------------------------

class TestModelSize:

    def test_parameter_count_under_100k(self, model):
        """v2 spec: ~50K parameters. Allow up to 100K for flexibility."""
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert n_params < 100_000, f"Model has {n_params} params — exceeds 100K limit"

    def test_parameter_count_over_10k(self, model):
        """Model should have meaningful capacity — at least 10K params."""
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert n_params > 10_000, f"Model has only {n_params} params — suspiciously small"


# ---------------------------------------------------------------------------
# Feature column alignment
# ---------------------------------------------------------------------------

class TestFeatureAlignment:

    def test_n_cat_matches_cardinalities(self):
        assert len(CAT_FEATURE_INDICES) == len(CAT_CARDINALITIES)

    def test_cat_and_cont_indices_cover_all_features(self):
        all_indices = sorted(CAT_FEATURE_INDICES + CONT_FEATURE_INDICES)
        assert all_indices == list(range(len(MODEL_FEATURE_COLUMNS)))

    def test_cat_indices_are_first(self):
        """Categorical features must be the first N columns in MODEL_FEATURE_COLUMNS."""
        assert CAT_FEATURE_INDICES == [0, 1, 2]