import numpy as np

from src.data import sample_source_xy, sample_target_x
from src.t5 import (
    DomainClassifier,
    fit_estimated_wcp,
    make_domain_data,
    run_estimated_wcp_once,
    run_t5_repeats,
    summarize_t5,
)


def test_make_domain_data_labels_source_and_target():
    x_source = np.array([-1.0, 0.0])
    x_target = np.array([1.0, 2.0, 3.0])

    x, y = make_domain_data(x_source, x_target)

    assert x.shape == (5, 1)
    assert np.array_equal(y, np.array([0, 0, 1, 1, 1]))


def test_domain_classifier_outputs_valid_probabilities_and_weights():
    rng = np.random.default_rng(0)
    x_source, _ = sample_source_xy(500, rng)
    x_target = sample_target_x("S1", 500, rng)

    classifier = DomainClassifier(clip=0.01)
    classifier.fit(x_source, x_target)

    prob = classifier.predict_target_prob(np.array([-1.0, 0.0, 1.0]))
    weights = classifier.density_ratio(np.array([-1.0, 0.0, 1.0]))

    assert prob.shape == (3,)
    assert weights.shape == (3,)
    assert np.all((prob > 0) & (prob < 1))
    assert np.all(weights > 0)
    assert classifier.auc(x_source, x_target) > 0.5


def test_fit_estimated_wcp_returns_interval_arrays():
    config = {
        "alpha": 0.1,
        "n_fit": 80,
        "n_cal": 60,
        "n_target_u": 70,
        "n_test": 90,
        "model_degree": 3,
        "seed": 0,
    }

    result = fit_estimated_wcp(config, seed=0, scenario="S1")

    assert result["q_hat"].shape == (config["n_test"],)
    assert result["lower"].shape == (config["n_test"],)
    assert result["upper"].shape == (config["n_test"],)
    assert result["weights_cal_hat"].shape == (config["n_cal"],)
    assert result["weights_test_hat"].shape == (config["n_test"],)
    assert np.all(result["weights_cal_hat"] > 0)


def test_run_estimated_wcp_once_returns_metrics():
    config = {
        "alpha": 0.1,
        "n_fit": 80,
        "n_cal": 60,
        "n_target_u": 70,
        "n_test": 90,
        "model_degree": 3,
        "seed": 0,
    }

    metrics = run_estimated_wcp_once(config, seed=0, scenario="S1")

    assert metrics["method"] == "estimated_wcp"
    assert 0 <= metrics["coverage"] <= 1
    assert metrics["mean_length"] > 0
    assert metrics["ess"] > 0
    assert metrics["max_weight"] > 0
    assert 0 <= metrics["domain_auc"] <= 1


def test_run_and_summarize_t5_repeats():
    config = {
        "alpha": 0.1,
        "n_fit": 80,
        "n_cal": 60,
        "n_target_u": 70,
        "n_test": 90,
        "model_degree": 3,
        "seed": 0,
    }

    rows = run_t5_repeats(config, n_repeats=2)
    summary = summarize_t5(rows)

    assert len(rows) == 6
    assert {row["method"] for row in rows} == {
        "split_cp",
        "oracle_wcp",
        "estimated_wcp",
    }
    assert {row["method"] for row in summary} == {
        "split_cp",
        "oracle_wcp",
        "estimated_wcp",
    }
    assert all("coverage_mean" in row for row in summary)
    assert all("mean_length" in row for row in summary)
