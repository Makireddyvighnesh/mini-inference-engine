from minillm_l4.benchmarks.commands.run_kv_capacity import (
    FirstFitContiguous,
    Request,
    request_mix,
    simulate_churn,
    static_capacity,
)


def test_first_fit_contiguous_exposes_external_fragmentation() -> None:
    memory = FirstFitContiguous(10)
    assert memory.allocate("a", 4) and memory.allocate("b", 2) and memory.allocate("c", 4)
    memory.release("a")
    memory.release("c")

    assert memory.free_slots == 8
    assert memory.largest_hole == 4
    assert not memory.allocate("d", 6)  # enough total space, no single hole
    memory.release("b")
    assert memory.largest_hole == 10 and memory.allocate("d", 6)


def test_static_capacity_counts_block_rounding_and_reservation_policies() -> None:
    requests = [Request(f"r{i}", prompt_tokens=10, output_tokens=11) for i in range(10)]  # 20 final

    result = static_capacity(requests, capacity_slots=64, block_sizes=[8, 16])

    assert result["contiguous_final_length"]["requests_admitted"] == 3
    assert result["contiguous_max_length"]["requests_admitted"] == 3
    assert result["paged_b8"]["requests_admitted"] == 2  # 24 slots each
    assert result["paged_b8"]["wasted_token_slots"] == 8
    assert result["paged_b16"]["requests_admitted"] == 2  # 32 slots each
    assert result["paged_b16"]["internal_waste_fraction"] == 12 / 32


def test_churn_finishes_every_request_under_both_policies() -> None:
    requests = request_mix(30, seed=3, shapes=((20, 5), (60, 10), (100, 3)))

    paged = simulate_churn(requests, capacity_slots=256, policy="paged", block_size=16)
    contiguous = simulate_churn(requests, capacity_slots=256, policy="contiguous")

    for row in (paged, contiguous):
        assert row["decode_steps"] > 0
        assert 0 < row["mean_used_budget_fraction"] <= row["mean_reserved_budget_fraction"] <= 1
    assert paged["fragmentation_blocked_steps"] == 0
