"""Module for training a reward model from the generated feedback."""

import math
import os
import functools
from typing import Union

import torch
import wandb
import gymnasium as gym
from pytorch_lightning import Callback, LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader, random_split

from multi_type_feedback.datatypes import FeedbackType
from multi_type_feedback.feedback_dataset import FeedbackDataset, LoadFeedbackDataset
from multi_type_feedback.networks import (
    LightningPolicyCnnNetwork,
    LightningPolicyNetwork,
    calculate_cpl_loss,
    calculate_single_reward_loss,
)
from multi_type_feedback.utils import TrainingUtils
from multi_type_feedback.networks import ResetActorOptimizerCallback


# for convenice sake, todo: make dynamic in the future
discount_factors = {
    "HalfCheetah-v5": 0.98,
    "Hopper-v5": 0.99,
    "Swimmer-v5": 0.9999,
    "Ant-v5": 0.99,
    "Walker2d-v5": 0.99,
    "ALE/BeamRider-v5": 0.99,
    "ALE/MsPacman-v5": 0.99,
    "ALE/Enduro-v5": 0.99,
    "ALE/Pong-v5": 0.99,
    "Humanoid-v5": 0.99,
    "highway-fast-v0": 0.8,
    "merge-v0": 0.8,
    "roundabout-v0": 0.8,
    "metaworld-sweep-into-v2": 0.99,
    "metaworld-button-press-v2": 0.99,
    "metaworld-pick-place-v2": 0.99,
}

# Utilize Tensor Cores of NVIDIA GPUs
torch.set_float32_matmul_precision("high")


def train_policy(
    args,
    policy_model: LightningModule,
    agent_model_id: str,
    feedback_type: FeedbackType,
    dataset: FeedbackDataset,
    maximum_epochs: int = 100,
    cpu_count: int = 4,
    algorithm: str = "sac",
    environment: str = "HalfCheetah-v3",
    gradient_clip_value: Union[float, None] = None,
    split_ratio: float = 0.8,
    enable_progress_bar=True,
    callback: Union[Callback, None] = None,
    num_ensemble_models: int = 4,
    noise_level: float = 0.0,
    n_feedback: int = -1,
    seed: int = 0,
    wandb_project_name: str = "multi-type-rlhf",
    save_path: str = "agents",
):
    """Train a reward model given trajectories data."""
    training_set_size = math.floor(split_ratio * len(dataset))
    train_set, val_set = random_split(
        dataset, lengths=[training_set_size, len(dataset) - training_set_size]
    )

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        # num_workers=cpu_count,
        num_workers=0,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        pin_memory=True,
        num_workers=1,
        drop_last=True,
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=save_path,
        filename=agent_model_id,
        monitor="val_loss",
    )

    # initialise the wandb logger and name your wandb project
    wandb_logger = WandbLogger(
        project=wandb_project_name,
        name=agent_model_id,
        config={
            "feedback_type": feedback_type,
            "noise_level": noise_level,
            "seed": seed,
            "environment": environment,
            "n_feedback": n_feedback,
            **vars(args),  # Include all args from argparse
        },
    )

    trainer = Trainer(
        max_epochs=maximum_epochs,
        accelerator="auto",
        devices="auto",
        log_every_n_steps=5,
        gradient_clip_val=gradient_clip_value,
        enable_progress_bar=enable_progress_bar,
        logger=wandb_logger,
        # accumulate_grad_batches=32, # NOTE: this is not compatible with manual optimization
        callbacks=[
            # EarlyStopping(monitor="val_loss", mode="min", patience=5),
            checkpoint_callback,
            ResetActorOptimizerCallback(bc_steps=args.bc_steps, learning_rate=1e-4),
            *([callback] if callback is not None else []),
        ],
    )
    trainer.fit(policy_model, train_loader, val_loader)
    
    # Policy evaluation
    print("Starting policy evaluation...")
    
    # Create environment for evaluation
    eval_env = gym.make(environment)
    
    # Set the policy model to evaluation mode
    policy_model.eval()
    
    # Evaluation parameters
    n_eval_episodes = 10
    episode_rewards = []
    episode_lengths = []

    print(f"Evaluating policy for {n_eval_episodes} episodes...")

    for episode in range(n_eval_episodes):
        obs, _ = eval_env.reset()
        episode_reward = 0
        episode_length = 0

        while True:
            # Convert observation to tensor
            obs_tensor = torch.as_tensor(obs, device=TrainingUtils.get_device(), dtype=torch.float32).unsqueeze(0)

            # Get action from policy (deterministic for evaluation)
            with torch.no_grad():
                action = policy_model.predict(obs_tensor) # This should output actions
                # If the model outputs logits for discrete actions, sample from them
                if isinstance(eval_env.action_space, gym.spaces.Discrete):
                    action_probs = torch.softmax(action, dim=-1)
                    action = torch.argmax(action_probs, dim=-1).cpu().numpy()[0]
                else:
                    # For continuous actions, use the output directly
                    action = action.cpu().numpy()[0]
            
            # Take action in environment
            obs, reward, terminated, truncated, info = eval_env.step(action)
            episode_reward += reward
            episode_length += 1
            
            if terminated or truncated:
                break
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        if (episode + 1) % 5 == 0:
            print(f"Episode {episode + 1}/{n_eval_episodes}, Reward: {episode_reward:.2f}")
    
    # Calculate statistics
    mean_reward = sum(episode_rewards) / len(episode_rewards)
    std_reward = (sum((r - mean_reward) ** 2 for r in episode_rewards) / len(episode_rewards)) ** 0.5
    mean_length = sum(episode_lengths) / len(episode_lengths)
    
    print(f"\nEvaluation Results:")
    print(f"Mean Reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    print(f"Mean Episode Length: {mean_length:.2f}")
    print(f"Episodes: {n_eval_episodes}")
    
    # Log to wandb if available
    if wandb.run is not None:
        wandb.log({
            "eval/mean_reward": mean_reward,
            "eval/std_reward": std_reward,
            "eval/mean_episode_length": mean_length,
            "eval/episodes": n_eval_episodes,
        })
    
    # Close the environment
    eval_env.close()
    
    # Save evaluation results
    eval_results_path = os.path.join(save_path, f"eval_{agent_model_id}_results.txt")
    with open(eval_results_path, "w") as f:
        f.write(f"Environment: {environment}\n")
        f.write(f"Algorithm: {algorithm}\n")
        f.write(f"Mean Reward: {mean_reward:.2f}\n")
        f.write(f"Std Reward: {std_reward:.2f}\n")
        f.write(f"Mean Episode Length: {mean_length:.2f}\n")
        f.write(f"Episodes: {n_eval_episodes}\n")
    
    print(f"Evaluation results saved to: {eval_results_path}")

    wandb.finish()

    return policy_model


def main():
    parser = TrainingUtils.setup_base_parser()
    parser.add_argument(
        "--alg", type=str, default="cpl"
    )
    parser.add_argument(
        "--feedback-type", type=str, default="comparative", help="Type of feedback"
    )
    parser.add_argument(
        "--n-ensemble", type=int, default=1, help="Number of ensemble models"
    )
    parser.add_argument(
        "--no-loading-bar", action="store_true", help="Disable loading bar"
    )
    parser.add_argument(
        "--feedback-folder",
        type=str,
        default="feedback",
        help="Folder to load feedback from",
    )
    parser.add_argument(
        "--save-folder",
        type=str,
        default="agents",
        help="Save folder for trained reward models",
    )
    parser.add_argument(
        "--max_epochs", type=int, default=1000,
    )
    
    parser.add_argument(
        "--alpha", type=float, default=0.1,
    )
    parser.add_argument(
        "--bc_coeff", type=float, default=0.0,
    )
    parser.add_argument(
        "--bc_data", type=str, default='all',
    )
    parser.add_argument(
        "--bc_steps", type=int, default=200000,
    )
    parser.add_argument(
        "--contrastive_bias", type=float, default=0.5,
    )
    parser.add_argument(
        "--batch_size", type=int, default=96,
    )
    parser.add_argument(
        "--last_activation", action="store_false",
    ) # very important
    parser.add_argument(
        "--normalization", action="store_false",
    ) # very important
    parser.add_argument(
        "--dropout_rate", type=float, default=0.0,
    )
    parser.add_argument(
        "--RM_alg", type=str, default="cpl", help="Reward model learns from the exp of which RL algorithm",
    )
    parser.add_argument(
        "--cuda_num", type=str,
    )
    args = parser.parse_args()
    args.RM_alg = args.algorithm
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_num

    TrainingUtils.set_seeds(args.seed)
    environment = TrainingUtils.setup_environment(
        args.environment, save_reset_wrapper=False
    )
    feedback_id, model_id = TrainingUtils.get_model_ids(args)
    
    # Setup reward model
    policy_model = (
        LightningPolicyCnnNetwork
        if "procgen" in args.environment or "ALE" in args.environment
        else LightningPolicyNetwork
    )(
        input_spaces=(environment.observation_space, environment.action_space),
        actor_layer_dims=[512, 512],
        encoder_layer_dims=[256, 256],
        bc_steps=args.bc_steps,
        action_hidden_dim=(
            16 if "procgen" in args.environment or "ALE" in args.environment else 32
        ),
        cnn_channels=(
            (16, 32, 32)
            if "procgen" in args.environment or "ALE" in args.environment
            else None
        ),
        output_dim=environment.action_space.shape[0],
        loss_function=functools.partial(
            calculate_cpl_loss, 
            alpha=args.alpha, 
            bc_data=args.bc_data,
            bc_steps=args.bc_steps,
            bc_coeff=args.bc_coeff,
            contrastive_bias=args.contrastive_bias
        ),
        learning_rate=1e-4,
        last_activation=torch.nn.Tanh if args.last_activation else None,
        normalization=torch.nn.LayerNorm if args.normalization else None,
        dropout_rate=args.dropout_rate,
        ensemble_count=args.n_ensemble,
    )

    dataset = LoadFeedbackDataset(
        os.path.join(args.feedback_folder, f"{feedback_id}.pkl"),
        args.feedback_type,
        args.n_feedback,
        noise_level=args.noise_level,
        env=environment if args.feedback_type == "demonstrative" else None,
        env_name=args.environment,
        seed=args.seed,
        discount_factor=discount_factors.get(
            args.environment, 0.99
        ),  # adapt for custom envs
    )

    train_policy(
        args,
        policy_model,
        model_id,
        args.feedback_type,
        dataset,
        algorithm=args.algorithm,
        maximum_epochs=args.max_epochs,
        split_ratio=1.0,
        environment=args.environment,
        cpu_count=os.cpu_count() or 8,
        num_ensemble_models=args.n_ensemble,
        enable_progress_bar=not args.no_loading_bar,
        noise_level=args.noise_level,
        n_feedback=args.n_feedback,
        seed=args.seed,
        wandb_project_name=args.wandb_project_name,
        save_path=args.save_folder,
    )


if __name__ == "__main__":
    main()
