import pytest
import torch
import numpy as np
from objects_gpu import Player, Game, Category
from unittest.mock import patch

@pytest.fixture
def setup_game():
    """Setup a test game with 2 players and 4 categories."""
    categories = {
        f"Q{i+1}": Category(
            name=f"Q{i+1}",
            mu=5.0 * (1.2 if i < 2 else 1.0),  # Higher mu for Q1/Q2
            sigma=2.0 * (1.6 if i % 2 == 0 else 1.0),  # Higher sigma for Q1/Q3
            size=30,
            log_or_normal="normal"
        )
        for i in range(4)
    }
    
    players = [
        Player(win_value=0.4, blind=False, level=100, name="Player_0"),
        Player(win_value=0.6, blind=True, level=100, name="Player_1")
    ]
    
    game = Game(
        num_players=2,
        to_admit=15,
        players=players,
        categories=categories,
        game_mode_type="top_k",
        top_k=5,
        log_normal=False
    )
    
    return game, players[0]  # Test with Player_0

def test_top_k_br_model_loading(setup_game):
    """Test if the model loads correctly."""
    game, player = setup_game
    feasible_numbers = {f"Q{i+1}": 10 for i in range(4)}
    
    with patch("torch.jit.load") as mock_load:
        mock_load.return_value.eval.return_value = None
        player.top_k_br(game, feasible_numbers)
        mock_load.assert_called_once_with("runs/topk_experiment/best_model_biased.pt.jit")

def test_top_k_br_output_shape(setup_game):
    """Test if the strategy output has correct shape and constraints."""
    game, player = setup_game
    feasible_numbers = {f"Q{i+1}": 10 for i in range(4)}
    
    player.top_k_br(game, feasible_numbers)
    
    # Check strategy format
    assert set(player.strategy.keys()) == {"Q1", "Q2", "Q3", "Q4"}
    assert all(0 <= v <= 1 for v in player.strategy.values())
    assert sum(player.strategy.values()) <= 1.0 + 1e-6  # Allow floating-point tolerance

def test_top_k_br_with_mock_model(setup_game):
    """Test with a mock model that returns predictable outputs."""
    game, player = setup_game
    feasible_numbers = {f"Q{i+1}": 10 for i in range(4)}
    
    # Mock model that prefers Q1
    class MockModel:
        def eval(self):
            return self
        def __call__(self, x, k):
            return torch.tensor([[10.0 if x[0, 0, 0] > 5 else 1.0]])  # Higher score for Q1
    
    with patch("torch.jit.load", return_value=MockModel()):
        player.top_k_br(game, feasible_numbers)
        assert player.strategy["Q1"] > player.strategy["Q3"]  # Q1 should be favored

def test_top_k_br_feasibility(setup_game):
    """Test if the strategy respects feasible numbers and prioritizes higher-mean categories."""
    game, player = setup_game
    feasible_numbers = {"Q1": 5, "Q2": 5, "Q3": 20, "Q4": 20}
    player.top_k_br(game, feasible_numbers)
    allocated = {k: v * game.categories[k].size for k, v in player.strategy.items()}

    # No category should exceed its feasible number
    for k in feasible_numbers:
        assert allocated[k] <= feasible_numbers[k] + 1e-6

    # Total allocation should not exceed to_admit
    assert sum(allocated.values()) <= game.to_admit + 1e-6

    # Q1 and Q2 should be filled first (since they have higher means)
    assert allocated["Q1"] >= feasible_numbers["Q1"] - 1e-6
    assert allocated["Q2"] >= feasible_numbers["Q2"] - 1e-6

    # Q4 should get zero allocation (since Q3 and Q4 have lower means and Q3 comes first)
    assert allocated["Q4"] <= 1e-6
