"""Post-processing helpers for binary fake/real classification."""


def map_probability_to_label(p_fake, threshold=0.5):
    """Map a fake-class probability to a human-readable label."""
    return "Fake" if p_fake >= threshold else "Real"
