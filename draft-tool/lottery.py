import random

def run_lottery(team_names: list[str], base_weights: list[float]) -> list[str]:
    """
    Run a weighted carry-over draft lottery.

    Args:
        team_names:   List of team names in the same order as base_weights.
        base_weights: Base weight for each team (higher = worse finish).

    Returns:
        Ordered list of team names representing their draft pick order.
    """
    remaining = list(zip(team_names, base_weights))
    carries = {name: 1.0 for name in team_names}
    pick_order = []

    while remaining:
        effective = [(name, bw * carries[name]) for name, bw in remaining]
        total = sum(ew for _, ew in effective)
        probs = [(name, ew / total) for name, ew in effective]

        names = [name for name, _ in probs]
        weights = [p for _, p in probs]
        winner = random.choices(names, weights=weights, k=1)[0]
        pick_order.append(winner)

        winner_prob = dict(probs)[winner]
        remaining = [(name, bw) for name, bw in remaining if name != winner]

        for name, _ in remaining:
            carries[name] += dict(probs)[name]

        _ = winner_prob  # carry update handled above

    return pick_order


if __name__ == "__main__":
    teams = [
        ("10th place", 4),
        ("9th place",  3),
        ("8th place",  2),
        ("7th place",  1),
    ]
    team_names = [t[0] for t in teams]
    base_weights = [t[1] for t in teams]

    print("Running draft lottery...\n")
    order = run_lottery(team_names, base_weights)
    for pick_num, team in enumerate(order, start=1):
        print(f"  Pick {pick_num}: {team}")
