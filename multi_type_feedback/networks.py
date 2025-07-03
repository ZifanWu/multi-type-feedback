"""Module for instantiating a neural network."""

# pylint: disable=arguments-differ
from typing import Callable, Tuple, Type, Union
import itertools
from utils import create_optim_groups
from schedules import linear_decay, linear_warmup

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from masksembles.torch import Masksembles1D, Masksembles2D
from pytorch_lightning import LightningModule
from torch import Tensor, dropout_, nn
from torch.nn.functional import mse_loss, nll_loss
from pytorch_lightning.callbacks import Callback

# Loss functions
single_reward_loss = nn.MSELoss()


def calculate_mse_loss(network: LightningModule, batch: Tensor):
    """Calculate the mean squared error loss for the reward."""
    return mse_loss(network(batch[0]), batch[1].unsqueeze(1), reduction="sum")


def calculate_mle_loss(network: LightningModule, batch: Tensor):
    """Calculate the maximum likelihood loss for the better trajectory."""
    rewards1 = network(batch[0]).flatten()
    rewards2 = network(batch[1]).flatten()

    probs_softmax = torch.exp(rewards1) / (torch.exp(rewards1) + torch.exp(rewards2))

    loss = -torch.sum(torch.log(probs_softmax))

    return loss


def calculate_pairwise_loss(network: LightningModule, batch: Tensor):
    """Calculate the maximum likelihood loss for the better trajectory."""
    (
        pair_obs,
        pair_actions,
    ), preferred_indices = batch  # preferred_indices: (batch_size,)

    # Unpack observations and actions for both trajectories
    obs1, obs2 = pair_obs[0], pair_actions[0]
    actions1, actions2 = pair_obs[1], pair_actions[1]
    # print(obs1.shape, actions1.shape) [Batch, Len, Dim]

    # Compute network outputs
    outputs1 = network(
        obs1, actions1
    )  # Shape: (batch_size, segment_length, output_dim)
    outputs2 = network(obs2, actions2)

    # Sum over sequence dimension
    rewards1 = outputs1.sum(dim=1).squeeze(-1) # Shape: (batch_size,)
    rewards2 = outputs2.sum(dim=1).squeeze(-1)

    # Stack rewards and compute log softmax
    rewards = torch.stack([rewards1, rewards2], dim=1)  # Shape: (batch_size, 2)
    log_probs = F.log_softmax(rewards, dim=1)

    # Compute NLL loss
    loss = nll_loss(log_probs, preferred_indices)

    return loss


def calculate_single_reward_loss(network: LightningModule, batch: Tensor):
    """Calculate the MSE loss between prediction and actual reward."""
    (observations, actions), targets = batch
    # Network output: (batch_size, segment_length, output_dim)
    outputs = network(observations, actions)

    # Sum over the sequence dimension to get total rewards per segment
    total_rewards = outputs.sum(dim=1)  # Shape: (batch_size, output_dim)

    # Ensure targets have the correct shape
    targets = targets.float().unsqueeze(1)  # Shape: (batch_size, 1)

    # Compute loss
    loss = single_reward_loss(total_rewards, targets)

    return loss


def calculate_cpl_loss(
    network: LightningModule, 
    batch: Tensor, 
    step: int,
    alpha: float=0.1, 
    bc_data: str='all',
    bc_steps: int=0,
    bc_coeff: float=0.0,
    contrastive_bias: float=1.0,
    ):

    (
        pair_obs,
        pair_actions,
    ), preferred_indices = batch  # preferred_indices: (batch_size,) #TODO check why the preferred indices always 1

    # Unpack observations and actions for both trajectories
    obs1, obs2 = pair_obs[0], pair_actions[0] # [Batch, Len, O_Dim]
    # print(obs1.shape, preferred_indices.shape) # [B, L, O_dim]
    actions1, actions2 = pair_obs[1], pair_actions[1] # [Batch, Len, A_Dim]
    if bc_data == 'pos': # In some cases we might want to only BC the positive data.
        pass
    else:
        obs = torch.cat((obs1, obs2), dim=0).reshape(-1, obs1.shape[-2], obs1.shape[-1])
        action = torch.cat((actions1, actions2), dim=0).reshape(-1, obs1.shape[-2], actions1.shape[-1])

    # Step 1: Compute the BC Loss from the log probabilities.
    dist = network(obs) # (B*2, L, A_dim)
    assert dist.shape == action.shape
    # For independent gaussian with unit var, logprob reduces to MSE.
    lp = -torch.square(dist - action).sum(dim=-1)
    bc_loss = (-lp).mean()

    # Step 2: Compute the advantages.
    adv = alpha * lp
    segment_adv = adv.sum(dim=-1)

    # Step 3: Compute the Loss.
    adv1, adv2 = torch.chunk(segment_adv, 2, dim=0)
    cpl_loss, accuracy = biased_bce_with_logits(adv1, adv2, preferred_indices.float(), bias=contrastive_bias)

    # Step 4: Combine CPL loss and BC loss
    if step < bc_steps:
        loss = bc_loss
    else:
        loss = cpl_loss + bc_coeff * bc_loss

    return loss


def biased_bce_with_logits(adv1, adv2, y, bias=1.0):
    # Apply the log-sum-exp trick.
    # y = 1 if we prefer x2 to x1
    # We need to implement the numerical stability trick.

    logit21 = adv2 - bias * adv1
    logit12 = adv1 - bias * adv2
    max21 = torch.clamp(-logit21, min=0, max=None)
    max12 = torch.clamp(-logit12, min=0, max=None)
    nlp21 = torch.log(torch.exp(-max21) + torch.exp(-logit21 - max21)) + max21
    nlp12 = torch.log(torch.exp(-max12) + torch.exp(-logit12 - max12)) + max12
    loss = y * nlp21 + (1 - y) * nlp12
    loss = loss.mean()

    # Now compute the accuracy
    with torch.no_grad():
        accuracy = ((adv2 > adv1) == torch.round(y)).float().mean()

    return loss, accuracy


def biased_bce_with_scores(adv, scores, bias=1.0):
    # For now label clip does nothing.
    # Could try doing this asymetric with two sides, but found that it doesn't work well.

    idx = torch.argsort(scores, dim=0)
    adv_sorted = adv[idx]

    # Compute normalized loss
    logits = adv_sorted.unsqueeze(0) - bias * adv_sorted.unsqueeze(1)
    max_val = torch.clamp(-logits, min=0, max=None)
    loss = torch.log(torch.exp(-max_val) + torch.exp(-logits - max_val)) + max_val

    loss = torch.triu(loss, diagonal=1)
    mask = loss != 0.0
    loss = loss.sum() / mask.sum()

    with torch.no_grad():
        unbiased_logits = adv_sorted.unsqueeze(0) - adv_sorted.unsqueeze(1)
        accuracy = ((unbiased_logits > 0) * mask).sum() / mask.sum()

    return loss, accuracy


# Lightning networks


class LightningPolicyNetwork(LightningModule):
    """Neural network to model the RL agent's reward using Pytorch Lightning."""

    def __init__(
        self,
        input_spaces: Tuple[gym.spaces.Space, gym.spaces.Space],
        actor_layer_dims: list[int],
        encoder_layer_dims: list[int],
        bc_steps: int,
        output_dim: int,
        action_hidden_dim: int,  # not used here
        loss_function: Callable[[LightningModule, Tensor], Tensor],
        learning_rate: float,
        cnn_channels: list[int] = None,  # not used, just for compatability
        activation_function: Type[nn.Module] = nn.ReLU,
        last_activation: Union[Type[nn.Module], None] = None,
        ensemble_count: int = 0,
        dropout_rate: float = 0.25,
        normalization: Type[nn.Module] = torch.nn.LayerNorm,
    ):
        super().__init__()

        self.loss_function = loss_function
        self.learning_rate = learning_rate
        self.ensemble_count = ensemble_count
        obs_space, action_space = input_spaces

        action_is_discrete = isinstance(action_space, gym.spaces.Discrete)

        input_dim = np.prod(obs_space.shape)

        # Initialize the encoder network
        encoder_layers_unit = [input_dim] + encoder_layer_dims
        encoder_layers: list[nn.Module] = []

        for idx in range(len(encoder_layers_unit) - 1):
            encoder_layers.append(nn.Linear(encoder_layers_unit[idx], encoder_layers_unit[idx + 1]))
            encoder_layers.append(activation_function())
        self.encoder = nn.Sequential(*encoder_layers)

        # Initialize the actor network
        actor_layers_unit = [encoder_layer_dims[-1]] + actor_layer_dims
        actor_layers: list[nn.Module] = []

        for idx in range(len(actor_layers_unit) - 1):
            actor_layers.append(nn.Linear(actor_layers_unit[idx], actor_layers_unit[idx + 1]))
            if dropout_rate > 0.0:
                actor_layers.append(nn.Dropout(dropout_rate))
            if normalization is not None:
                actor_layers.append(normalization(actor_layers_unit[idx + 1]))
            actor_layers.append(activation_function())

        actor_layers.append(nn.Linear(actor_layers_unit[-1], output_dim))

        if last_activation is not None:
            actor_layers.append(last_activation())
        self.actor = nn.Sequential(*actor_layers)

        # Initialize the weights
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight)

                layer.bias.data.zero_()

        self.save_hyperparameters()
        
        # NOTE
        self.step = 0
        self.bc_steps = bc_steps
        self.schedulers = {}
        self.schedulers_class = {'actor': linear_warmup}
        self.automatic_optimization = False

    def forward(self, observations: Tensor):
        """Do a forward pass through the neural network (inference)."""
        # observations: (batch_size, segment_length, obs_dim)
        # actions: (batch_size, segment_length, action_dim)
        if (
            len(observations.shape) > 3
        ):  # needs to be done to support the 2d spaces of highway-env, probably better do in in env wrapper
            observations = observations.flatten(start_dim=2)
        batch_size, segment_length, obs_dim = observations.shape

        # Flatten the batch and sequence dimensions
        obs_flat = observations.reshape(-1, obs_dim)

        # Concatenate observations and actions
        batch = obs_flat

        # Pass through the network
        output = self.actor(self.encoder(batch))  # Shape: (batch_size * segment_length, output_dim)

        # Reshape back to (batch_size, segment_length, output_dim)
        output = output.reshape(batch_size, segment_length, -1)

        return output

    def predict(self, observations: Tensor):
        """Do a forward pass through the neural network (inference)."""
        return self.actor(self.encoder(observations.cpu()))

    def training_step(self, batch, batch_idx):
        # 获取 optimizers
        opt_encoder, opt_actor = self.optimizers()

        # 梯度清零
        opt_encoder.zero_grad()
        opt_actor.zero_grad()

        # 计算 loss
        loss = self.loss_function(self, batch, self.step)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        self.step += 1

        # 反向传播
        self.manual_backward(loss)

        # 两个 optimizer 都 step
        opt_encoder.step()
        opt_actor.step()

        return loss

    def validation_step(self, batch: Tensor, _batch_idx: int):
        """Compute the loss for validation."""
        loss = self.loss_function(self, batch, self.step)
        self.log("val_loss", loss, prog_bar=True)

    def configure_optimizers(self):
        """Configure optimizer and scheduler to optimize the neural network."""
        optimizer_encoder = torch.optim.Adam(self.encoder.parameters(), lr=self.learning_rate)
        optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=self.learning_rate)

        # Scheduler: 前bc_steps步actor的lr=1.0，之后线性warmup到1.0
        warmup_steps = 10000
        def actor_lr_lambda(step):
            if step < self.bc_steps:
                return 1.0
            else:
                # linear_warmup: multiplier * min(1.0, step / total_steps)
                # 这里step-bc_steps从0开始递增
                progress = (step - self.bc_steps) / warmup_steps
                return min(1.0, progress)

        scheduler_encoder = torch.optim.lr_scheduler.LambdaLR(optimizer_encoder, lr_lambda=lambda step: 1.0)
        scheduler_actor = torch.optim.lr_scheduler.LambdaLR(optimizer_actor, lr_lambda=actor_lr_lambda)

    #     return [optimizer_encoder, optimizer_actor], [scheduler_encoder, scheduler_actor]
        return (
            {"optimizer": optimizer_encoder, "lr_scheduler": scheduler_encoder},
            {"optimizer": optimizer_actor, "lr_scheduler": scheduler_actor}
                )

class LightningPolicyCnnNetwork(LightningModule):
    """Neural network to model the RL agent's reward using Pytorch Lightning,
    based on the Impapala CNN architecture. Use given layer_num, with a single
    fully connected layer"""

    def __init__(
        self,
        input_spaces: Tuple[gym.spaces.Space, gym.spaces.Space],
        layer_num: int,
        output_dim: int,
        hidden_dim: int,  # not used here
        action_hidden_dim: int,
        loss_function: Callable[[LightningModule, Tensor], Tensor],
        learning_rate: float,
        cnn_channels: list[int] = (16, 32, 32),
        activation_function: Type[nn.Module] = nn.ReLU,
        last_activation: Union[Type[nn.Module], None] = None,
        ensemble_count: int = 0,
    ):
        super().__init__()

        self.loss_function = loss_function
        self.learning_rate = learning_rate
        self.ensemble_count = ensemble_count
        obs_space, action_space = input_spaces
        input_channels = obs_space.shape[0]

        # Initialize the network
        layers = []
        for i in range(layer_num):
            layers.append(self.conv_sequence(input_channels, cnn_channels[i]))
            input_channels = cnn_channels[i]

        self.conv_layers = nn.Sequential(*layers)
        self.flatten = nn.Flatten()

        # action input_layer
        action_shape = action_space.shape if action_space.shape else 1
        self.action_in = nn.Linear(action_shape, action_hidden_dim)
        self.masksemble_out = Masksembles1D(
            channels=action_hidden_dim, n=self.ensemble_count, scale=1.8
        ).float()

        self.fc = nn.Linear(
            self.compute_flattened_size(obs_space.shape, cnn_channels)
            + action_hidden_dim,
            output_dim,
        )

        self.save_hyperparameters()

    def conv_layer(self, in_channels, out_channels):
        return nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def residual_block(self, in_channels):
        return nn.Sequential(
            nn.ReLU(),
            Masksembles2D(
                channels=in_channels, n=self.ensemble_count, scale=1.8
            ).float(),
            self.conv_layer(in_channels, in_channels),
            nn.ReLU(),
            Masksembles2D(
                channels=in_channels, n=self.ensemble_count, scale=1.8
            ).float(),
            self.conv_layer(in_channels, in_channels),
        )

    def conv_sequence(self, in_channels, out_channels):
        return nn.Sequential(
            self.conv_layer(in_channels, out_channels),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            self.residual_block(out_channels),
            self.residual_block(out_channels),
        )

    def compute_flattened_size(self, observation_space, cnn_channels):
        with torch.no_grad():
            sample_input = torch.zeros(self.ensemble_count, *observation_space).squeeze(
                -1
            )
            sample_output = self.conv_layers(sample_input).flatten(start_dim=1)
            return sample_output.shape[-1]

    def forward(self, observations, actions):
        # observations: (batch_size, segment_length, channels, height, width)
        # actions: (batch_size, segment_length, action_dim)
        batch_size, segment_length, channels, height, width = observations.shape
        _, _, action_dim = actions.shape

        # Process observations through convolutional layers
        obs_flat = observations.reshape(
            batch_size * segment_length, channels, height, width
        )
        x = self.conv_layers(obs_flat)
        x = self.flatten(x)
        x = F.relu(x)

        # Flatten actions
        actions_flat = actions.reshape(-1, action_dim)
        act = self.action_in(actions_flat)
        act = self.masksemble_out(act)
        act = F.relu(act)

        # Concatenate processed observations and actions
        output = torch.cat((x, act), dim=1)
        output = self.fc(output)  # Shape: (batch_size * segment_length, output_dim)

        # Reshape back to (batch_size, segment_length, output_dim)
        output = output.reshape(batch_size, segment_length, -1)

        return output

    def training_step(self, batch: Tensor):
        """Compute the loss for training."""
        loss = self.loss_function(self, batch)
        self.log("train_loss", loss, prog_bar=True)

        return loss

    def validation_step(self, batch: Tensor, _batch_idx: int):
        """Compute the loss for validation."""
        loss = self.loss_function(self, batch)
        self.log("val_loss", loss, prog_bar=True)

    def configure_optimizers(self):
        """Configure optimizer to optimize the neural network."""
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer


class LightningNetwork(LightningModule):
    """Neural network to model the RL agent's reward using Pytorch Lightning."""

    def __init__(
        self,
        input_spaces: Tuple[gym.spaces.Space, gym.spaces.Space],
        layer_num: int,
        output_dim: int,
        hidden_dim: int,
        action_hidden_dim: int,  # not used here
        loss_function: Callable[[LightningModule, Tensor], Tensor],
        learning_rate: float,
        cnn_channels: list[int] = None,  # not used, just for compatability
        activation_function: Type[nn.Module] = nn.ReLU,
        last_activation: Union[Type[nn.Module], None] = None,
        ensemble_count: int = 0,
    ):
        super().__init__()

        self.loss_function = loss_function
        self.learning_rate = learning_rate
        self.ensemble_count = ensemble_count
        obs_space, action_space = input_spaces

        action_is_discrete = isinstance(action_space, gym.spaces.Discrete)

        input_dim = np.prod(obs_space.shape) + (
            np.prod(action_space.shape) if not action_is_discrete else action_space.n
        )

        # Initialize the network
        layers_unit = [input_dim] + [hidden_dim] * (layer_num - 1)

        layers: list[nn.Module] = []

        for idx in range(len(layers_unit) - 1):
            layers.append(nn.Linear(layers_unit[idx], layers_unit[idx + 1]))
            layers.append(activation_function())

            if self.ensemble_count > 1:
                layers.append(
                    Masksembles1D(
                        channels=layers_unit[idx + 1],
                        n=self.ensemble_count,
                        scale=1.8,
                    ).float()
                )

        layers.append(nn.Linear(layers_unit[-1], output_dim))

        if last_activation is not None:
            layers.append(last_activation())

            if self.ensemble_count > 1:
                layers.append(
                    Masksembles1D(
                        channels=output_dim, n=self.ensemble_count, scale=1.8
                    ).float()
                )

        self.network = nn.Sequential(*layers)

        # Initialize the weights
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight)

                layer.bias.data.zero_()

        self.save_hyperparameters()

    def forward(self, observations: Tensor, actions: Tensor):
        """Do a forward pass through the neural network (inference)."""
        # observations: (batch_size, segment_length, obs_dim)
        # actions: (batch_size, segment_length, action_dim)

        if (
            len(observations.shape) > 3
        ):  # needs to be done to support the 2d spaces of highway-env, probably better do in in env wrapper
            observations = observations.flatten(start_dim=2)

        batch_size, segment_length, obs_dim = observations.shape
        _, _, action_dim = actions.shape

        # Flatten the batch and sequence dimensions
        obs_flat = observations.reshape(-1, obs_dim)
        actions_flat = actions.reshape(-1, action_dim)

        # Concatenate observations and actions
        batch = torch.cat(
            (obs_flat, actions_flat), dim=1
        )  # Shape: (batch_size * segment_length, obs_dim + action_dim)

        # Pass through the network
        output = self.network(batch)  # Shape: (batch_size * segment_length, output_dim)

        # Reshape back to (batch_size, segment_length, output_dim)
        output = output.reshape(batch_size, segment_length, -1)

        return output

    def training_step(self, batch: Tensor):
        """Compute the loss for training."""
        loss = self.loss_function(self, batch)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch: Tensor, _batch_idx: int):
        """Compute the loss for validation."""
        loss = self.loss_function(self, batch)
        self.log("val_loss", loss, prog_bar=True)

    def configure_optimizers(self):
        """Configure optimizer to optimize the neural network."""
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        return optimizer


class LightningCnnNetwork(LightningModule):
    """Neural network to model the RL agent's reward using Pytorch Lightning,
    based on the Impapala CNN architecture. Use given layer_num, with a single
    fully connected layer"""

    def __init__(
        self,
        input_spaces: Tuple[gym.spaces.Space, gym.spaces.Space],
        layer_num: int,
        output_dim: int,
        hidden_dim: int,  # not used here
        action_hidden_dim: int,
        loss_function: Callable[[LightningModule, Tensor], Tensor],
        learning_rate: float,
        cnn_channels: list[int] = (16, 32, 32),
        activation_function: Type[nn.Module] = nn.ReLU,
        last_activation: Union[Type[nn.Module], None] = None,
        ensemble_count: int = 0,
    ):
        super().__init__()

        self.loss_function = loss_function
        self.learning_rate = learning_rate
        self.ensemble_count = ensemble_count
        obs_space, action_space = input_spaces
        input_channels = obs_space.shape[0]

        # Initialize the network
        layers = []
        for i in range(layer_num):
            layers.append(self.conv_sequence(input_channels, cnn_channels[i]))
            input_channels = cnn_channels[i]

        self.conv_layers = nn.Sequential(*layers)
        self.flatten = nn.Flatten()

        # action input_layer
        action_shape = action_space.shape if action_space.shape else 1
        self.action_in = nn.Linear(action_shape, action_hidden_dim)
        self.masksemble_out = Masksembles1D(
            channels=action_hidden_dim, n=self.ensemble_count, scale=1.8
        ).float()

        self.fc = nn.Linear(
            self.compute_flattened_size(obs_space.shape, cnn_channels)
            + action_hidden_dim,
            output_dim,
        )

        self.save_hyperparameters()

    def conv_layer(self, in_channels, out_channels):
        return nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def residual_block(self, in_channels):
        return nn.Sequential(
            nn.ReLU(),
            Masksembles2D(
                channels=in_channels, n=self.ensemble_count, scale=1.8
            ).float(),
            self.conv_layer(in_channels, in_channels),
            nn.ReLU(),
            Masksembles2D(
                channels=in_channels, n=self.ensemble_count, scale=1.8
            ).float(),
            self.conv_layer(in_channels, in_channels),
        )

    def conv_sequence(self, in_channels, out_channels):
        return nn.Sequential(
            self.conv_layer(in_channels, out_channels),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            self.residual_block(out_channels),
            self.residual_block(out_channels),
        )

    def compute_flattened_size(self, observation_space, cnn_channels):
        with torch.no_grad():
            sample_input = torch.zeros(self.ensemble_count, *observation_space).squeeze(
                -1
            )
            sample_output = self.conv_layers(sample_input).flatten(start_dim=1)
            return sample_output.shape[-1]

    def forward(self, observations, actions):
        # observations: (batch_size, segment_length, channels, height, width)
        # actions: (batch_size, segment_length, action_dim)
        batch_size, segment_length, channels, height, width = observations.shape
        _, _, action_dim = actions.shape

        # Process observations through convolutional layers
        obs_flat = observations.reshape(
            batch_size * segment_length, channels, height, width
        )
        x = self.conv_layers(obs_flat)
        x = self.flatten(x)
        x = F.relu(x)

        # Flatten actions
        actions_flat = actions.reshape(-1, action_dim)
        act = self.action_in(actions_flat)
        act = self.masksemble_out(act)
        act = F.relu(act)

        # Concatenate processed observations and actions
        output = torch.cat((x, act), dim=1)
        output = self.fc(output)  # Shape: (batch_size * segment_length, output_dim)

        # Reshape back to (batch_size, segment_length, output_dim)
        output = output.reshape(batch_size, segment_length, -1)

        return output

    def training_step(self, batch: Tensor):
        """Compute the loss for training."""
        loss = self.loss_function(self, batch)
        self.log("train_loss", loss, prog_bar=True)

        return loss

    def validation_step(self, batch: Tensor, _batch_idx: int):
        """Compute the loss for validation."""
        loss = self.loss_function(self, batch)
        self.log("val_loss", loss, prog_bar=True)

    def configure_optimizers(self):
        """Configure optimizer to optimize the neural network."""
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        return optimizer


class ResetActorOptimizerCallback(Callback):
    def __init__(self, bc_steps, learning_rate):
        super().__init__()
        self.bc_steps = bc_steps
        self.learning_rate = learning_rate
        self.reset_done = False

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if not self.reset_done and pl_module.global_step == self.bc_steps: 
            # NOTE: global_step is the optimizer step of the whole training process (including both the encoder and the actor)
            # 重新创建 actor optimizer
            new_actor_optimizer = torch.optim.Adam(pl_module.actor.parameters(), lr=self.learning_rate)
            # 替换 trainer.optimizers 里的 actor optimizer（假设actor在第二个位置）
            trainer.optimizers[1] = new_actor_optimizer
            # 如果有 scheduler 也可以重置
            # trainer.lr_schedulers[1]['scheduler'] = ...
            self.reset_done = True
