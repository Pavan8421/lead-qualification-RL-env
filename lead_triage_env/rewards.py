"""Per-step reward shaping (Phase 2 design)."""


def step_reward(
    *,
    outcome: str,
    action: str,
    quality: str,
    tier_waste_mult: float,
) -> float:
    """
    Dense reward signal. Wasted high-touch on low-quality leads adds extra penalty.
    """
    if outcome == "ignored_terminal":
        return 0.0

    base = 0.0
    if outcome == "converted":
        base = 10.0
    elif outcome == "positive_reply":
        base = 3.0
    elif outcome == "no_response":
        base = -1.0
    elif outcome == "churned":
        base = -4.0
    elif outcome == "horizon":
        base = -1.5

    step_cost = -0.15
    if outcome == "horizon":
        return base + step_cost

    waste = 0.0
    if quality == "low" and action == "CALL" and outcome in ("no_response", "churned"):
        waste = -2.0 * tier_waste_mult
    if quality == "low" and action == "EMAIL" and outcome in ("no_response", "churned"):
        waste = -0.5 * tier_waste_mult

    return base + waste + step_cost


def ignore_opportunity_cost(quality: str) -> float:
    """Extra penalty when IGNORE is chosen on higher-value latent leads."""
    if quality == "high":
        return -1.5
    if quality == "mid":
        return -0.5
    return 0.05
