"""Multi-agent environments: Predator-Prey, Coin Game, Simple Spread."""

from src.environments.env_base import MultiAgentEnv
from src.environments.predator_prey import PredatorPreyEnv
from src.environments.coin_game import CoinGameEnv
from src.environments.simple_spread import SimpleSpreadEnv

__all__ = [
    "MultiAgentEnv",
    "PredatorPreyEnv",
    "CoinGameEnv",
    "SimpleSpreadEnv",
]

ENV_REGISTRY: dict[str, type[MultiAgentEnv]] = {
    "predator_prey": PredatorPreyEnv,
    "coin_game": CoinGameEnv,
    "simple_spread": SimpleSpreadEnv,
}
