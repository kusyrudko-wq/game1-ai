"""Zero-sum DQN self-play. No solver, expert labels, or table initialization."""
from collections import deque
from dataclasses import asdict, dataclass
import math
from pathlib import Path
import random

import torch
from torch import nn

from .game import Position
from .neural_agent import NeuralAgent, QNetwork, encode_board


@dataclass(frozen=True)
class DQNConfig:
    episodes: int = 20000
    seed: int = 42
    learning_rate: float = 0.001
    gamma: float = 1.0
    epsilon_start: float = 1.0
    epsilon_end: float = 0.1
    epsilon_decay_episodes: int = 16000
    replay_capacity: int = 20000
    batch_size: int = 128
    warmup_steps: int = 512
    train_every: int = 4
    target_update: int = 250  # gradient updates, not moves
    gradient_clip: float = 10.0
    report_every: int = 1000
    threads: int = 1
    width: int = 64

    def __post_init__(self):
        for name in ('episodes', 'epsilon_decay_episodes', 'replay_capacity',
                     'batch_size', 'warmup_steps', 'train_every', 'target_update',
                     'report_every', 'threads', 'width'):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f'{name} must be a positive integer.')
        if self.batch_size > self.replay_capacity or self.warmup_steps > self.replay_capacity:
            raise ValueError('Replay capacity must cover batch size and warmup.')
        if not 0 <= self.gamma <= 1:
            raise ValueError('gamma must be between 0 and 1.')
        if not 0 <= self.epsilon_end <= self.epsilon_start <= 1:
            raise ValueError('Require 0 <= epsilon_end <= epsilon_start <= 1.')
        for name in ('learning_rate', 'gradient_clip'):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f'{name} must be finite and positive.')


def transition(position, move):
    """Store both states in their respective current-player perspectives."""
    next_position = position.play(move)
    result = next_position.result()
    return (tuple(encode_board(position).tolist()), move,
            float(result * position.player) if result is not None else 0.0,
            tuple(encode_board(next_position).tolist()), result is not None,
            tuple(i in next_position.legal_moves() for i in range(9)))


@torch.no_grad()
def bellman_targets(target_network, next_states, rewards, terminals, legal, gamma):
    """Terminal rows never bootstrap, including wins with empty squares."""
    targets = rewards.clone()
    active = ~terminals
    if active.any():
        if not legal[active].any(dim=1).all():
            raise ValueError('Nonterminal next states need a legal action.')
        values = target_network(next_states[active]).masked_fill(~legal[active], -torch.inf)
        targets[active] = -gamma * values.max(dim=1).values
    return targets


class DQNTrainer:
    """CPU trainer; checkpoints are taken only between complete games."""

    def __init__(self, config=DQNConfig()):
        self.config = config
        torch.set_num_threads(config.threads)
        self.rng = random.Random(config.seed)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(config.seed)
            self.network = QNetwork(config.width)
            self.target = QNetwork(config.width)
        self.target.load_state_dict(self.network.state_dict())
        self.target.eval().requires_grad_(False)
        self.agent = NeuralAgent(self.network)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=config.learning_rate)
        self.replay = deque(maxlen=config.replay_capacity)
        self.episode = self.steps = self.updates = 0
        self.history = []
        self.counts = {1: 0, -1: 0, 0: 0}
        self.loss_sum = 0.0
        self.loss_count = 0

    def epsilon(self):
        fraction = min(1.0, self.episode / max(1, self.config.epsilon_decay_episodes - 1))
        return self.config.epsilon_start + fraction * (self.config.epsilon_end - self.config.epsilon_start)

    def choose(self, position, epsilon):
        if not 0 <= epsilon <= 1:
            raise ValueError('epsilon must be between 0 and 1.')
        moves = position.legal_moves()
        if not moves:
            raise ValueError('Cannot choose a move in a finished game.')
        return self.rng.choice(moves) if self.rng.random() < epsilon else self.agent.choose(position)

    def update(self):
        if len(self.replay) < max(self.config.batch_size, self.config.warmup_steps):
            return None
        batch = self.rng.sample(list(self.replay), self.config.batch_size)
        states, moves, rewards, next_states, terminals, legal = zip(*batch)
        states = torch.tensor(states, dtype=torch.float32)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        targets = bellman_targets(self.target, next_states,
                                 torch.tensor(rewards, dtype=torch.float32),
                                 torch.tensor(terminals, dtype=torch.bool),
                                 torch.tensor(legal, dtype=torch.bool), self.config.gamma)
        self.network.train()
        predictions = self.network(states).gather(1, torch.tensor(moves)[:, None]).squeeze(1)
        loss = nn.functional.smooth_l1_loss(predictions, targets)
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite training loss.')
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), self.config.gradient_clip)
        self.optimizer.step()
        self.updates += 1
        if self.updates % self.config.target_update == 0:
            self.target.load_state_dict(self.network.state_dict())
        self.network.eval()
        return loss.item()

    def run(self, until=None, callback=None):
        end = self.config.episodes if until is None else until
        if not self.episode <= end <= self.config.episodes:
            raise ValueError('Requested episode is outside the training budget.')
        while self.episode < end:
            epsilon = self.epsilon()
            position = Position()
            while position.result() is None:
                move = self.choose(position, epsilon)
                self.replay.append(transition(position, move))
                position = position.play(move)
                self.steps += 1
                if self.steps % self.config.train_every == 0:
                    loss = self.update()
                    if loss is not None:
                        self.loss_sum += loss
                        self.loss_count += 1
            self.episode += 1
            self.counts[position.result()] += 1
            if self.episode % self.config.report_every == 0 or self.episode == self.config.episodes:
                row = dict(episode=self.episode, steps=self.steps, updates=self.updates,
                           epsilon=epsilon, replay_size=len(self.replay),
                           mean_loss=self.loss_sum / self.loss_count if self.loss_count else None,
                           x_wins=self.counts[1], o_wins=self.counts[-1], draws=self.counts[0])
                self.history.append(row)
                self.counts = {1: 0, -1: 0, 0: 0}
                self.loss_sum = 0.0
                self.loss_count = 0
                if callback:
                    callback(self, row)
        return self.agent, self.history

    def save_checkpoint(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(format_version=1, algorithm='zero-sum-dqn', config=asdict(self.config),
                       network=self.network.state_dict(), target=self.target.state_dict(),
                       optimizer=self.optimizer.state_dict(), replay=list(self.replay),
                       rng=self.rng.getstate(), episode=self.episode, steps=self.steps,
                       updates=self.updates, history=self.history, counts=self.counts,
                       loss_sum=self.loss_sum, loss_count=self.loss_count)
        with path.open('xb') as file:
            torch.save(payload, file)

    @classmethod
    def load_checkpoint(cls, path):
        payload = torch.load(path, map_location='cpu', weights_only=True)
        if payload.get('format_version') != 1 or payload.get('algorithm') != 'zero-sum-dqn':
            raise ValueError('Unsupported DQN checkpoint.')
        trainer = cls(DQNConfig(**payload['config']))
        trainer.network.load_state_dict(payload['network'])
        trainer.target.load_state_dict(payload['target'])
        trainer.optimizer.load_state_dict(payload['optimizer'])
        trainer.replay.extend(payload['replay'])
        trainer.rng.setstate(payload['rng'])
        for name in ('episode', 'steps', 'updates', 'history', 'counts', 'loss_sum', 'loss_count'):
            setattr(trainer, name, payload[name])
        return trainer


def save_weights(agent, path, metadata=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as file:
        torch.save(dict(format_version=1, algorithm='zero-sum-dqn',
                        network=agent.network.state_dict(), width=agent.network.width, metadata=metadata or {}), file)


def load_agent(path):
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if payload.get('format_version') != 1 or payload.get('algorithm') != 'zero-sum-dqn':
        raise ValueError('Unsupported DQN model.')
    # Old artifacts have no width field; infer it from their saved first layer.
    width = payload['network']['layers.0.weight'].shape[0]
    network = QNetwork(width)
    network.load_state_dict(payload['network'])
    return NeuralAgent(network)

