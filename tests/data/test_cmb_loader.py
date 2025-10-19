import numpy as np

from dataio.loaders import load_cmb_priors


def test_load_cmb_priors_planck2018():
    priors = load_cmb_priors("planck2018")
    labels = priors["labels"]
    cov = priors["cov"]
    means = priors["means"]
    assert isinstance(labels, list) and labels
    assert cov.shape[0] == cov.shape[1] == len(labels)
    eigvals = np.linalg.eigvalsh(cov)
    assert np.all(eigvals > 0.0)
    for label in labels:
        assert label in means
    meta = priors["meta"]
    assert "source" in meta
    assert meta.get("source_tag") in {"primary", "sample"}


def test_planck2018_loader_shape():
    priors = load_cmb_priors("planck2018")
    cov = priors["cov"]
    labels = priors["labels"]
    assert cov.shape[0] == cov.shape[1] == len(labels)
    assert "R" in priors["means"]
    assert cov[0, 0] > 0.0
