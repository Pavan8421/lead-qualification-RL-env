from lead_triage_env.rewards import reward_breakdown, step_reward


def test_reward_breakdown_matches_legacy_scalar_for_outcome_core() -> None:
    breakdown = reward_breakdown(
        outcome="positive_reply",
        action="EMAIL",
        quality="mid",
        tier_waste_mult=1.0,
        converted=False,
        step_index=1,
        max_steps=4,
        action_parsed=True,
        legal_actions=["CALL", "EMAIL", "IGNORE"],
        repetition_penalty=0.0,
        contact_attempts=1,
        max_contacts=None,
        terminal_grader=0.0,
    )
    legacy = step_reward(
        outcome="positive_reply",
        action="EMAIL",
        quality="mid",
        tier_waste_mult=1.0,
    )
    assert breakdown.outcome_reward == legacy


def test_reward_breakdown_total_is_sum_of_columns() -> None:
    breakdown = reward_breakdown(
        outcome="converted",
        action="CALL",
        quality="high",
        tier_waste_mult=1.0,
        converted=True,
        step_index=1,
        max_steps=4,
        action_parsed=True,
        legal_actions=["CALL", "EMAIL", "FOLLOW_UP", "IGNORE"],
        repetition_penalty=-0.25,
        contact_attempts=2,
        max_contacts=3,
        terminal_grader=0.8,
    )
    expected = (
        breakdown.outcome_reward
        + breakdown.efficiency_reward
        + breakdown.format_compliance
        + breakdown.repetition_penalty
        + breakdown.budget_compliance
        + breakdown.terminal_grader
    )
    assert breakdown.total == expected
