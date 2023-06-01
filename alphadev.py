# Copyright 2023 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Pseudocode description of the AlphaDev algorithm."""

###########################
########## Content ########
# 1. Environment
# 2. Networks
#   2.1 Network helpers
#   2.2 Representation network
#   2.3 Prediction network (correctness and latency values and policy)
# 3. Helpers
# 4. Part 1: Self-Play
# 5. Part 2: Training
###########################

import collections
import functools
import math
from typing import Any, Callable, Dict, NamedTuple, Optional, Sequence

import chex
import haiku as hk
import jax
import jax.lax
import jax.numpy as jnp
import ml_collections
import numpy
import optax


############################
###### 1. Environment ######


class TaskSpec(NamedTuple):
  max_program_size: int
  num_inputs: int
  num_funcs: int
  num_locations: int
  num_actions: int
  correct_reward: float
  correctness_reward_weight: float
  latency_reward_weight: float
  latency_quantile: float


class AssemblyGame(object):
  """The environment AlphaDev is interacting with."""

  class AssemblyInstruction(object):
    pass

  class AssemblySimulator(object):

    # pylint: disable-next=unused-argument
    def apply(self, instruction):
      return {}

    def measure_latency(self, program) -> float:
      pass

  def __init__(self, task_spec):
    self.task_spec = task_spec
    self.program = []
    self.simulator = self.AssemblySimulator(task_spec)
    self.previous_correct_items = 0

  def step(self, action):
    instruction = self.AssemblyInstruction(action)
    self.program.append(instruction)
    self.execution_state = self.simulator.apply(instruction)
    return self.observation(), self.correctness_reward()

  def observation(self):
    return {
        'program': self.program,
        'program_length': len(self.program),
        'memory': self.execution_state.memory,
        'registers': self.execution_state.registers,
    }

  def correctness_reward(self) -> float:
    """Computes a reward based on the correctness of the output."""
    make_expected_outputs = lambda: []
    expected_outputs = make_expected_outputs()
    state = self.execution_state

    # Weighted sum of correctly placed items
    correct_items = 0
    for output, expected in zip(state.memory, expected_outputs):
      correct_items += output.weight * sum(
          output[i] == expected[i] for i in range(len(output))
      )
    reward = self.task_spec.correctness_reward_weight * (
        correct_items - self.previous_correct_items
    )
    self.previous_correct_items = correct_items

    # Bonus for fully correct programs
    all_correct = all(
        output == expected
        for output, expected in zip(state.memory, expected_outputs)
    )
    reward += self.task_spec.correct_reward * all_correct

    return reward

  def latency_reward(self) -> float:
    latency_samples = [
        self.simulator.measure_latency(self.program)
        for _ in range(self.task_spec.num_latency_simulation)
    ]
    return (
        numpy.quantile(latency_samples, self.task_spec.latency_quantile)
        * self.task_spec.latency_reward_weight
    )

  def clone(self):
    pass


######## End Environment ########
#################################

#####################################
############ 2. Networks ############

######## 2.1 Network helpers ########


class Action(object):
  """Action representation."""

  def __init__(self, index: int):
    self.index = index

  def __hash__(self):
    return self.index

  def __eq__(self, other):
    return self.index == other.index

  def __gt__(self, other):
    return self.index > other.index


class NetworkOutput(NamedTuple):
  value: float
  correctness_value_logits: jnp.ndarray
  latency_value_logits: jnp.ndarray
  policy_logits: Dict[Action, float]


class Network(object):
  """Wrapper around Representation and Prediction networks."""

  def __init__(self, hparams: ml_collections.ConfigDict, task_spec: TaskSpec):
    self.representation = hk.transform(RepresentationNet(
        hparams, task_spec, hparams.embedding_dim
    ))
    self.prediction = hk.transform(PredictionNet(
        task_spec=task_spec,
        value_max=hparams.value.max,
        value_num_bins=hparams.value.num_bins,
        embedding_dim=hparams.embedding_dim,
    ))
    rep_key, pred_key = jax.random.PRNGKey(42).split()
    self.params = {
        'representation': self.representation.init(rep_key),
        'prediction': self.prediction.init(pred_key),
    }

  def inference(self, params: Any, observation: jnp.array) -> NetworkOutput:
    # representation + prediction function
    embedding = self.representation.apply(params['representation'], observation)
    return self.prediction.apply(params['prediction'], embedding)

  def get_params(self):
    # Returns the weights of this network.
    return self.params

  def update_params(self, updates: Any) -> None:
    # Update network weights internally.
    self.params = jax.tree_map(lambda p, u: p + u, self.params, updates)

  def training_steps(self) -> int:
    # How many steps / batches the network has been trained for.
    return 0


class UniformNetwork(object):
  """Network representation that returns uniform output."""

  # pylint: disable-next=unused-argument
  def inference(self, observation) -> NetworkOutput:
    # representation + prediction function
    return NetworkOutput(0, 0, 0, {})

  def get_params(self):
    # Returns the weights of this network.
    return self.params

  def update_params(self, updates: Any) -> None:
    # Update network weights internally.
    self.params = jax.tree_map(lambda p, u: p + u, self.params, updates)

  def training_steps(self) -> int:
    # How many steps / batches the network has been trained for.
    return 0


######## 2.2 Representation Network ########


class MultiQueryAttentionBlock:
  """Attention with multiple query heads and a single shared key and value head.

  Implementation of "Fast Transformer Decoding: One Write-Head is All You Need",
  see https://arxiv.org/abs/1911.02150.
  """


class ResBlockV2:
  """Layer-normed variant of the block from https://arxiv.org/abs/1603.05027."""


def int2bin(integers_array: jnp.array) -> jnp.array:
  """Converts an array of integers to an array of its 32bit representation bits.

  Conversion goes from array of shape (S1, S2, ..., SN) to (S1, S2, ..., SN*32),
  i.e. all binary arrays are concatenated. Also note that the single 32-long
  binary sequences are reversed, i.e. the number 1 will be converted to the
  binary 1000000... . This is irrelevant for ML problems.

  Args:
    integers_array: array of integers to convert.

  Returns:
    array of bits (on or off) in boolean type.
  """
  flat_arr = integers_array.astype(jnp.int32).reshape(-1, 1)
  bin_mask = jnp.tile(2 ** jnp.arange(32), (flat_arr.shape[0], 1))
  return ((flat_arr & bin_mask) != 0).reshape(
      *integers_array.shape[:-1], integers_array.shape[-1] * 32
  )


def bin2int(binary_array: jnp.array) -> jnp.array:
  """Reverses operation of int2bin."""
  u_binary_array = binary_array.reshape(
      *binary_array.shape[:-1], binary_array.shape[-1] // 32, 32
  )
  exp = jnp.tile(2 ** jnp.arange(32), u_binary_array.shape[:-1] + (1,))
  return jnp.sum(exp * u_binary_array, axis=-1)


class RepresentationNet(hk.Module):
  """Representation network."""

  def __init__(
      self,
      hparams: ml_collections.ConfigDict,
      task_spec: TaskSpec,
      embedding_dim: int,
      name: str = 'representation',
  ):
    super().__init__(name=name)
    self._hparams = hparams
    self._task_spec = task_spec
    self._embedding_dim = embedding_dim

  def __call__(self, inputs):
    batch_size = inputs['program'].shape[0]

    program_encoding = None
    if self._hparams.representation.use_program:
      program_encoding = self._encode_program(inputs, batch_size)

    if (
        self._hparams.representation.use_locations
        and self._hparams.representation.use_locations_binary
    ):
      raise ValueError(
          'only one of `use_locations` and `use_locations_binary` may be used.'
      )
    locations_encoding = None
    if self._hparams.representation.use_locations:
      locations_encoding = self._make_locations_encoding_onehot(
          inputs, batch_size
      )
    elif self._hparams.representation.use_locations_binary:
      locations_encoding = self._make_locations_encoding_binary(
          inputs, batch_size
      )

    permutation_embedding = None
    if self._hparams.representation.use_permutation_embedding:
      permutation_embedding = self.make_permutation_embedding(batch_size)

    return self.aggregate_locations_program(
        locations_encoding, permutation_embedding, program_encoding, batch_size
    )

  def _encode_program(self, inputs, batch_size):
    program = inputs['program']
    max_program_size = inputs['program'].shape[1]
    program_length = inputs['program_length'].astype(jnp.int32)
    program_onehot = self.make_program_onehot(
        program, batch_size, max_program_size
    )
    program_encoding = self.apply_program_mlp_embedder(program_onehot)
    program_encoding = self.apply_program_attention_embedder(program_encoding)
    return self.pad_program_encoding(
        program_encoding, batch_size, program_length, max_program_size
    )

  def aggregate_locations_program(
      self,
      locations_encoding,
      unused_permutation_embedding,
      program_encoding,
      batch_size,
  ):
    locations_embedder = hk.Sequential(
        [
            hk.Linear(self._embedding_dim),
            hk.LayerNorm(axis=-1),
            jax.nn.relu,
            hk.Linear(self._embedding_dim),
        ],
        name='per_locations_embedder',
    )

    # locations_encoding.shape == [B, P, D] so map embedder across locations to
    # share weights
    locations_embedding = hk.vmap(
        locations_embedder, in_axes=1, out_axes=1, split_rng=False
    )(locations_encoding)

    program_encoded_repeat = self.repeat_program_encoding(
        program_encoding, batch_size
    )

    grouped_representation = jnp.concatenate(
        [locations_embedding, program_encoded_repeat], axis=-1
    )

    return self.apply_joint_embedder(grouped_representation, batch_size)

  def repeat_program_encoding(self, program_encoding, batch_size):
    return jnp.broadcast_to(
        program_encoding,
        [batch_size, self._task_spec.num_inputs, program_encoding.shape[-1]],
    )

  def apply_joint_embedder(self, grouped_representation, batch_size):
    all_locations_net = hk.Sequential(
        [
            hk.Linear(self._embedding_dim),
            hk.LayerNorm(axis=-1),
            jax.nn.relu,
            hk.Linear(self._embedding_dim),
        ],
        name='per_element_embedder',
    )
    joint_locations_net = hk.Sequential(
        [
            hk.Linear(self._embedding_dim),
            hk.LayerNorm(axis=-1),
            jax.nn.relu,
            hk.Linear(self._embedding_dim),
        ],
        name='joint_embedder',
    )
    joint_resnet = [
        ResBlockV2(self._embedding_dim, name=f'joint_resblock_{i}')
        for i in range(self._hparams.representation.repr_net_res_blocks)
    ]

    chex.assert_shape(
        grouped_representation, (batch_size, self._task_spec.num_inputs, None)
    )
    permutations_encoded = all_locations_net(grouped_representation)
    # Combine all permutations into a single vector.
    joint_encoding = joint_locations_net(jnp.mean(permutations_encoded, axis=1))
    for net in joint_resnet:
      joint_encoding = net(joint_encoding)
    return joint_encoding

  def make_program_onehot(self, program, batch_size, max_program_size):
    func = program[:, :, 0]
    arg1 = program[:, :, 1]
    arg2 = program[:, :, 2]
    func_onehot = jax.nn.one_hot(func, self._task_spec.num_funcs)
    arg1_onehot = jax.nn.one_hot(arg1, self._task_spec.num_locations)
    arg2_onehot = jax.nn.one_hot(arg2, self._task_spec.num_locations)
    program_onehot = jnp.concatenate(
        [func_onehot, arg1_onehot, arg2_onehot], axis=-1
    )
    chex.assert_shape(program_onehot, (batch_size, max_program_size, None))
    return program_onehot

  def pad_program_encoding(
      self, program_encoding, batch_size, program_length, max_program_size
  ):
    """Pads the program encoding to account for state-action stagger."""
    chex.assert_shape(program_encoding, (batch_size, max_program_size, None))

    empty_program_output = jnp.zeros(
        [batch_size, program_encoding.shape[-1]],
    )
    program_encoding = jnp.concatenate(
        [empty_program_output[:, None, :], program_encoding], axis=1
    )

    program_length_onehot = jax.nn.one_hot(program_length, max_program_size + 1)

    program_encoding = jnp.einsum(
        'bnd,bNn->bNd', program_encoding, program_length_onehot
    )

    return program_encoding

  def apply_program_mlp_embedder(self, program_encoding):
    program_embedder = hk.Sequential(
        [
            hk.Linear(self._embedding_dim),
            hk.LayerNorm(axis=-1),
            jax.nn.relu,
            hk.Linear(self._embedding_dim),
        ],
        name='per_instruction_program_embedder',
    )

    program_encoding = program_embedder(program_encoding)
    return program_encoding

  def apply_program_attention_embedder(self, program_encoding):
    attention_params = self._hparams.representation.attention
    make_attention_block = functools.partial(
        MultiQueryAttentionBlock, attention_params, causal_mask=False
    )
    attention_encoders = [
        make_attention_block(name=f'attention_program_sequencer_{i}')
        for i in range(self._hparams.representation.attention_num_layers)
    ]

    *_, seq_size, feat_size = program_encoding.shape

    position_encodings = jnp.broadcast_to(
        MultiQueryAttentionBlock.sinusoid_position_encoding(
            seq_size, feat_size
        ),
        program_encoding.shape,
    )
    program_encoding += position_encodings

    for e in attention_encoders:
      program_encoding, _ = e(program_encoding, encoded_state=None)

    return program_encoding

  def _make_locations_encoding_onehot(self, inputs, batch_size):
    """Creates location encoding using onehot representation."""
    memory = inputs['memory']
    registers = inputs['registers']
    locations = jnp.concatenate([memory, registers], axis=-1)  # [B, H, P, D]
    locations = jnp.transpose(locations, [0, 2, 1, 3])  # [B, P, H, D]

    # One-hot encode the values in the memory and average everything across
    # permutations.
    locations_onehot = jax.nn.one_hot(
        locations, self._task_spec.num_location_values, dtype=jnp.int32
    )
    locations_onehot = locations_onehot.reshape(
        [batch_size, self._task_spec.num_inputs, -1]
    )

    return locations_onehot

  def _make_locations_encoding_binary(self, inputs, batch_size):
    """Creates location encoding using binary representation."""

    memory_binary = int2bin(inputs['memory']).astype(jnp.float32)
    registers_binary = int2bin(inputs['registers']).astype(jnp.float32)
    # Note the extra I dimension for the length of the binary integer (32)
    locations = jnp.concatenate(
        [memory_binary, registers_binary], axis=-1
    )  # [B, H, P, D*I]
    locations = jnp.transpose(locations, [0, 2, 1, 3])  # [B, P, H, D*I]

    locations = locations.reshape([batch_size, self._task_spec.num_inputs, -1])

    return locations


######## 2.3 Prediction Network ########


def make_head_network(
    embedding_dim: int,
    output_size: int,
    num_hidden_layers: int = 2,
    name: Optional[str] = None,
) -> Callable[[jnp.ndarray,], jnp.ndarray]:
  return hk.Sequential(
      [ResBlockV2(embedding_dim) for _ in range(num_hidden_layers)]
      + [hk.Linear(output_size)],
      name=name,
  )


class DistributionSupport(object):

  def __init__(self, value_max: float, num_bins: int):
    self.value_max = value_max
    self.num_bins = num_bins

  def mean(self, logits: jnp.ndarray) -> float:
    pass

  def scalar_to_two_hot(self, scalar: float) -> jnp.ndarray:
    pass


class CategoricalHead(hk.Module):
  """A head that represents continuous values by a categorical distribution."""

  def __init__(
      self,
      embedding_dim: int,
      support: DistributionSupport,
      name: str = 'CategoricalHead',
  ):
    super().__init__(name=name)
    self._value_support = support
    self._embedding_dim = embedding_dim
    self._head = make_head_network(
        embedding_dim, output_size=self._value_support.num_bins
    )

  def __call__(self, x: jnp.ndarray):
    # For training returns the logits, for inference the mean.
    logits = self._head(x)
    probs = jax.nn.softmax(logits)
    mean = jax.vmap(self._value_support.mean)(probs)
    return dict(logits=logits, mean=mean)


class PredictionNet(hk.Module):
  """MuZero prediction network."""

  def __init__(
      self,
      task_spec: TaskSpec,
      value_max: float,
      value_num_bins: int,
      embedding_dim: int,
      name: str = 'prediction',
  ):
    super().__init__(name=name)
    self.task_spec = task_spec
    self.support = DistributionSupport(self.value_max, self.value_num_bins)
    self.embedding_dim = embedding_dim

  def __call__(self, embedding: jnp.ndarray):
    policy_head = make_head_network(
        self.embedding_dim, self.task_spec.num_actions
    )
    value_head = CategoricalHead(self.embedding_dim, self.support)
    latency_value_head = CategoricalHead(self.embedding_dim, self.support)
    correctness_value = value_head(embedding)
    latency_value = latency_value_head(embedding)

    return NetworkOutput(
        value=correctness_value['mean'] + latency_value['mean'],
        correctness_value_logits=correctness_value['logits'],
        latency_value_logits=latency_value['logits'],
        policy=policy_head(embedding),
    )


####### End Networks ########
#############################

#############################
####### 3. Helpers ##########

MAXIMUM_FLOAT_VALUE = float('inf')

KnownBounds = collections.namedtuple('KnownBounds', ['min', 'max'])


class AlphaDevConfig(object):
  """AlphaDev configuration."""

  def __init__(
      self,
  ):
    ### Self-Play
    self.num_actors = 128  # TPU actors
    # pylint: disable-next=g-long-lambda
    self.visit_softmax_temperature_fn = lambda steps: (
        1.0 if steps < 500e3 else 0.5 if steps < 750e3 else 0.25
    )
    self.max_moves = jnp.inf
    self.num_simulations = 800
    self.discount = 1.0

    # Root prior exploration noise.
    self.root_dirichlet_alpha = 0.03
    self.root_exploration_fraction = 0.25

    # UCB formula
    self.pb_c_base = 19652
    self.pb_c_init = 1.25

    self.known_bounds = KnownBounds(-6.0, 6.0)

    # Environment: spec of the Variable Sort 3 task
    self.task_spec = TaskSpec(
        max_program_size=100,
        num_inputs=17,
        num_funcs=14,
        num_locations=19,
        num_actions=271,
        correct_reward=1.0,
        correctness_reward_weight=2.0,
        latency_reward_weight=0.5,
        latency_quantile=0,
    )

    ### Network architecture
    self.hparams = ml_collections.ConfigDict()
    self.hparams.embedding_dim = 512
    self.hparams.representation = ml_collections.ConfigDict()
    self.hparams.representation.use_program = True
    self.hparams.representation.use_locations = True
    self.hparams.representation.use_locations_binary = False
    self.hparams.representation.use_permutation_embedding = False
    self.hparams.representation.repr_net_res_blocks = 8
    self.hparams.representation.attention = ml_collections.ConfigDict()
    self.hparams.representation.attention.head_depth = 128
    self.hparams.representation.attention.num_heads = 4
    self.hparams.representation.attention.attention_dropout = False
    self.hparams.representation.attention.position_encoding = 'absolute'
    self.hparams.representation.attention_num_layers = 6
    self.hparams.value = ml_collections.ConfigDict()
    self.hparams.value.max = 3.0  # These two parameters are task / reward-
    self.hparams.value.num_bins = 301  # dependent and need to be adjusted.

    ### Training
    self.training_steps = int(1000e3)
    self.checkpoint_interval = 500
    self.target_network_interval = 100
    self.window_size = int(1e6)
    self.batch_size = 512
    self.td_steps = 5
    self.lr_init = 2e-4
    self.momentum = 0.9

  def new_game(self):
    return Game(self.task_spec.num_actions, self.discount, self.task_spec)


class MinMaxStats(object):
  """A class that holds the min-max values of the tree."""

  def __init__(self, known_bounds: Optional[KnownBounds]):
    self.maximum = known_bounds.max if known_bounds else -MAXIMUM_FLOAT_VALUE
    self.minimum = known_bounds.min if known_bounds else MAXIMUM_FLOAT_VALUE

  def update(self, value: float):
    self.maximum = max(self.maximum, value)
    self.minimum = min(self.minimum, value)

  def normalize(self, value: float) -> float:
    if self.maximum > self.minimum:
      # We normalize only when we have set the maximum and minimum values.
      return (value - self.minimum) / (self.maximum - self.minimum)
    return value


class Player(object):
  pass


class Node(object):
  """MCTS node."""

  def __init__(self, prior: float):
    self.visit_count = 0
    self.to_play = -1
    self.prior = prior
    self.value_sum = 0
    self.children = {}
    self.hidden_state = None
    self.reward = 0

  def expanded(self) -> bool:
    return bool(self.children)

  def value(self) -> float:
    if self.visit_count == 0:
      return 0
    return self.value_sum / self.visit_count


class ActionHistory(object):
  """Simple history container used inside the search.

  Only used to keep track of the actions executed.
  """

  def __init__(self, history: Sequence[Action], action_space_size: int):
    self.history = list(history)
    self.action_space_size = action_space_size

  def clone(self):
    return ActionHistory(self.history, self.action_space_size)

  def add_action(self, action: Action):
    self.history.append(action)

  def last_action(self) -> Action:
    return self.history[-1]

  def action_space(self) -> Sequence[Action]:
    return [Action(i) for i in range(self.action_space_size)]

  def to_play(self) -> Player:
    return Player()


class Target(NamedTuple):
  correctness_value: float
  latency_value: float
  policy: Sequence[int]
  bootstrap_discount: float


class Sample(NamedTuple):
  observation: Dict[str, jnp.ndarray]
  bootstrap_observation: Dict[str, jnp.ndarray]
  target: Target


class Game(object):
  """A single episode of interaction with the environment."""

  def __init__(
      self, action_space_size: int, discount: float, task_spec: TaskSpec
  ):
    self.task_spec = task_spec
    self.environment = AssemblyGame(task_spec)
    self.history = []
    self.rewards = []
    self.latency_reward = 0
    self.child_visits = []
    self.root_values = []
    self.action_space_size = action_space_size
    self.discount = discount

  def terminal(self) -> bool:
    # Game specific termination rules.
    # For sorting, a game is terminal if we sort all sequences correctly or
    # we reached the end of the buffer.
    pass

  def is_correct(self) -> bool:
    # Whether the current algorithm solves the game.

  def legal_actions(self) -> Sequence[Action]:
    # Game specific calculation of legal actions.
    return []

  def apply(self, action: Action):
    _, reward = self.environment.step(action)
    self.rewards.append(reward)
    self.history.append(action)
    if self.terminal() and self.is_correct():
      self.latency_reward = self.environment.latency_reward()

  def store_search_statistics(self, root: Node):
    sum_visits = sum(child.visit_count for child in root.children.values())
    action_space = (Action(index) for index in range(self.action_space_size))
    self.child_visits.append(
        [
            root.children[a].visit_count / sum_visits
            if a in root.children
            else 0
            for a in action_space
        ]
    )
    self.root_values.append(root.value())

  def make_observation(self, state_index: int):
    if state_index == -1:
      return self.environment.observation()
    env = AssemblyGame(self.task_spec)
    for action in self.history[:state_index]:
      observation, _ = env.step(action)
    return observation

  def make_target(
      # pylint: disable-next=unused-argument
      self, state_index: int, td_steps: int, to_play: Player
  ) -> Target:
    """Creates the value target for training."""
    # The value target is the discounted sum of all rewards until N steps
    # into the future, to which we will add the discounted boostrapped future
    # value.
    bootstrap_index = state_index + td_steps

    for i, reward in enumerate(self.rewards[state_index:bootstrap_index]):
      value += reward * self.discount**i  # pytype: disable=unsupported-operands

    if bootstrap_index < len(self.root_values):
      bootstrap_discount = self.discount**td_steps
    else:
      bootstrap_discount = 0

    return Target(
        value,
        self.latency_reward,
        self.child_visits[state_index],
        bootstrap_discount,
    )

  def to_play(self) -> Player:
    return Player()

  def action_history(self) -> ActionHistory:
    return ActionHistory(self.history, self.action_space_size)


class ReplayBuffer(object):
  """Replay buffer object storing games for training."""

  def __init__(self, config: AlphaDevConfig):
    self.window_size = config.window_size
    self.batch_size = config.batch_size
    self.buffer = []

  def save_game(self, game):
    if len(self.buffer) > self.window_size:
      self.buffer.pop(0)
    self.buffer.append(game)

  def sample_batch(self, td_steps: int) -> Sequence[Sample]:
    games = [self.sample_game() for _ in range(self.batch_size)]
    game_pos = [(g, self.sample_position(g)) for g in games]
    # pylint: disable=g-complex-comprehension
    return [
        Sample(
            observation=g.make_observation(i),
            bootstrap_observation=g.make_observation(i + td_steps),
            target=g.make_target(i, td_steps, g.to_play()),
        )
        for (g, i) in game_pos
    ]
    # pylint: enable=g-complex-comprehension

  def sample_game(self) -> Game:
    # Sample game from buffer either uniformly or according to some priority.
    return self.buffer[0]

  # pylint: disable-next=unused-argument
  def sample_position(self, game) -> int:
    # Sample position from game either uniformly or according to some priority.
    return -1


class SharedStorage(object):
  """Controls which network is used at inference."""

  def __init__(self):
    self._networks = {}

  def latest_network(self) -> Network:
    if self._networks:
      return self._networks[max(self._networks.keys())]
    else:
      # policy -> uniform, value -> 0, reward -> 0
      return make_uniform_network()

  def save_network(self, step: int, network: Network):
    self._networks[step] = network


##### End Helpers ########
##########################


# AlphaDev training is split into two independent parts: Network training and
# self-play data generation.
# These two parts only communicate by transferring the latest network checkpoint
# from the training to the self-play, and the finished games from the self-play
# to the training.
def alphadev(config: AlphaDevConfig):
  storage = SharedStorage()
  replay_buffer = ReplayBuffer(config)

  for _ in range(config.num_actors):
    launch_job(run_selfplay, config, storage, replay_buffer)

  train_network(config, storage, replay_buffer)

  return storage.latest_network()


#####################################
####### 4. Part 1: Self-Play ########


# Each self-play job is independent of all others; it takes the latest network
# snapshot, produces a game and makes it available to the training job by
# writing it to a shared replay buffer.
def run_selfplay(
    config: AlphaDevConfig, storage: SharedStorage, replay_buffer: ReplayBuffer
):
  while True:
    network = storage.latest_network()
    game = play_game(config, network)
    replay_buffer.save_game(game)


def play_game(config: AlphaDevConfig, network: Network) -> Game:
  """Plays an AlphaDev game.

  Each game is produced by starting at the initial empty program, then
  repeatedly executing a Monte Carlo Tree Search to generate moves until the end
  of the game is reached.

  Args:
    config: An instance of the AlphaDev configuration.
    network: Networks used for inference.

  Returns:
    The played game.
  """

  game = config.new_game()

  while not game.terminal() and len(game.history) < config.max_moves:
    min_max_stats = MinMaxStats(config.known_bounds)

    # Initialisation of the root node and addition of exploration noise
    root = Node(0)
    current_observation = game.make_observation(-1)
    network_output = network.inference(current_observation)
    _expand_node(
        root, game.to_play(), game.legal_actions(), network_output, reward=0
    )
    _backpropagate(
        [root],
        network_output.value,
        game.to_play(),
        config.discount,
        min_max_stats,
    )
    _add_exploration_noise(config, root)

    # We then run a Monte Carlo Tree Search using the environment.
    run_mcts(
        config,
        root,
        game.action_history(),
        network,
        min_max_stats,
        game.environment,
    )
    action = _select_action(config, len(game.history), root, network)
    game.apply(action)
    game.store_search_statistics(root)
  return game


def run_mcts(
    config: AlphaDevConfig,
    root: Node,
    action_history: ActionHistory,
    network: Network,
    min_max_stats: MinMaxStats,
    env: AssemblyGame,
):
  """Runs the Monte Carlo Tree Search algorithm.

  To decide on an action, we run N simulations, always starting at the root of
  the search tree and traversing the tree according to the UCB formula until we
  reach a leaf node.

  Args:
    config: AlphaDev configuration
    root: The root node of the MCTS tree from which we start the algorithm
    action_history: history of the actions taken so far.
    network: instances of the networks that will be used.
    min_max_stats: min-max statistics for the tree.
    env: an instance of the AssemblyGame.
  """

  for _ in range(config.num_simulations):
    history = action_history.clone()
    node = root
    search_path = [node]
    sim_env = env.clone()

    while node.expanded():
      action, node = _select_child(config, node, min_max_stats)
      sim_env.step(action)
      history.add_action(action)
      search_path.append(node)

    # Inside the search tree we use the environment to obtain the next
    # observation and reward given an action.
    observation, reward = sim_env.step(action)
    network_output = network.inference(observation)
    _expand_node(
        node, history.to_play(), history.action_space(), network_output, reward
    )

    _backpropagate(
        search_path,
        network_output.value,
        history.to_play(),
        config.discount,
        min_max_stats,
    )


def _select_action(
    # pylint: disable-next=unused-argument
    config: AlphaDevConfig, num_moves: int, node: Node, network: Network
):
  visit_counts = [
      (child.visit_count, action) for action, child in node.children.items()
  ]
  t = config.visit_softmax_temperature_fn(
      training_steps=network.training_steps()
  )
  _, action = softmax_sample(visit_counts, t)
  return action


def _select_child(
    config: AlphaDevConfig, node: Node, min_max_stats: MinMaxStats
):
  """Selects the child with the highest UCB score."""
  _, action, child = max(
      (_ucb_score(config, node, child, min_max_stats), action, child)
      for action, child in node.children.items()
  )
  return action, child


def _ucb_score(
    config: AlphaDevConfig,
    parent: Node,
    child: Node,
    min_max_stats: MinMaxStats,
) -> float:
  """Computes the UCB score based on its value + exploration based on prior."""
  pb_c = (
      math.log((parent.visit_count + config.pb_c_base + 1) / config.pb_c_base)
      + config.pb_c_init
  )
  pb_c *= math.sqrt(parent.visit_count) / (child.visit_count + 1)

  prior_score = pb_c * child.prior
  if child.visit_count > 0:
    value_score = min_max_stats.normalize(
        child.reward + config.discount * child.value()
    )
  else:
    value_score = 0
  return prior_score + value_score


def _expand_node(
    node: Node,
    to_play: Player,
    actions: Sequence[Action],
    network_output: NetworkOutput,
    reward: float,
):
  """Expands the node using value, reward and policy predictions from the NN."""
  node.to_play = to_play
  node.hidden_state = network_output.hidden_state
  node.reward = reward
  policy = {a: math.exp(network_output.policy_logits[a]) for a in actions}
  policy_sum = sum(policy.values())
  for action, p in policy.items():
    node.children[action] = Node(p / policy_sum)


def _backpropagate(
    search_path: Sequence[Node],
    value: float,
    to_play: Player,
    discount: float,
    min_max_stats: MinMaxStats,
):
  """Propagates the evaluation all the way up the tree to the root."""
  for node in reversed(search_path):
    node.value_sum += value if node.to_play == to_play else -value
    node.visit_count += 1
    min_max_stats.update(node.value())

    value = node.reward + discount * value


def _add_exploration_noise(config: AlphaDevConfig, node: Node):
  """Adds dirichlet noise to the prior of the root to encourage exploration."""
  actions = list(node.children.keys())
  noise = numpy.random.dirichlet([config.root_dirichlet_alpha] * len(actions))
  frac = config.root_exploration_fraction
  for a, n in zip(actions, noise):
    node.children[a].prior = node.children[a].prior * (1 - frac) + n * frac


########### End Self-Play ###########
#####################################

#####################################
####### 5. Part 2: Training #########


def train_network(
    config: AlphaDevConfig, storage: SharedStorage, replay_buffer: ReplayBuffer
):
  """Trains the network on data stored in the replay buffer."""
  network = Network(config.hparams, config.task_spec)
  target_network = Network(config.hparams, config.task_spec)
  optimizer = optax.sgd(config.lr_init, config.momentum)
  optimizer_state = optimizer.init(network.get_params())

  for i in range(config.training_steps):
    if i % config.checkpoint_interval == 0:
      storage.save_network(i, network)
    if i % config.target_network_interval == 0:
      target_network = network.copy()
    batch = replay_buffer.sample_batch(config.num_unroll_steps, config.td_steps)
    optimizer_state = _update_weights(
        optimizer, optimizer_state, network, target_network, batch)
  storage.save_network(config.training_steps, network)


def scale_gradient(tensor: Any, scale):
  """Scales the gradient for the backward pass."""
  return tensor * scale + jax.lax.stop_gradient(tensor) * (1 - scale)


def _loss_fn(
    network_params: jnp.array,
    target_network_params: jnp.array,
    network: Network,
    target_network: Network,
    batch: Sequence[Sample]
) -> float:
  """Computes loss."""
  loss = 0
  for observation, bootstrap_obs, target in batch:
    predictions = network.inference(network_params, observation)
    bootstrap_predictions = target_network.inference(
        target_network_params, bootstrap_obs)
    target_correctness, target_latency, target_policy, bootstrap_discount = (
        target
    )
    target_correctness += (
        bootstrap_discount * bootstrap_predictions.correctness_value_logits
    )

    l = optax.softmax_cross_entropy(predictions.policy_logits, target_policy)
    l += scalar_loss(
        predictions.correctness_value_logits, target_correctness, network
    )
    l += scalar_loss(predictions.latency_value_logits, target_latency, network)
    loss += l
  loss /= len(batch)
  return loss


_loss_grad = jax.grad(_loss_fn, argnums=0)


def _update_weights(
    optimizer: optax.GradientTransformation,
    optimizer_state: Any,
    network: Network,
    target_network: Network,
    batch: Sequence[Sample],
) -> Any:
  """Updates the weight of the network."""
  updates = _loss_grad(
      network.get_params(),
      target_network.get_params(),
      network,
      target_network,
      batch)

  optim_updates, new_optim_state = optimizer.update(updates, optimizer_state)
  network.update_params(optim_updates)
  return new_optim_state


def scalar_loss(prediction, target, network) -> float:
  support = network.prediction.support
  return optax.softmax_cross_entropy(
      prediction, support.scalar_to_two_hot(target)
  )


######### End Training ###########
##################################

################################################################################
############################# End of pseudocode ################################
################################################################################


# Stubs to make the typechecker happy.
# pylint: disable-next=unused-argument
def softmax_sample(distribution, temperature: float):
  return 0, 0


def launch_job(f, *args):
  f(*args)


def make_uniform_network():
  return UniformNetwork()
