"""
finetune_KTO.py

Script for fine-tuning OpenVLA models using the Kahneman-Tversky Optimization (KTO) loss with continuous rewards
between 0 and 1.

Notes:
    - This implementation adjusts the KTO loss to handle continuous rewards.
    - Requires a dataset where each trajectory or step has an associated reward between 0 (undesirable) and 1 (desirable).
"""


import os
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import copy
import logging
import time
import datetime

import draccus
import torch
import torch.distributed as dist
import tqdm
from accelerate import PartialState
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
from transformers import AutoConfig, AutoImageProcessor
from transformers.modeling_outputs import CausalLMOutputWithPast
from torch.nn import functional as F

import wandb
from prismatic.models.backbones.llm.prompting import PurePromptBuilder, VicunaV15ChatPromptBuilder
from prismatic.util.data_utils import PaddedCollatorForActionPrediction
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.datasets import RLDSBatchTransform, RLDSDataset
from prismatic.vla.datasets.rlds.utils.data_utils import save_dataset_statistics

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor

# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - Rank %(message)s'
)


@dataclass
class FinetuneConfig:
    vla_path: str = "openvla/openvla-7b"  # Path to OpenVLA model (on HuggingFace Hub)
    
    # Directory Paths
    data_root_dir: Path = Path("tensorflow_datasets/fanta_single_short_ft1_dataset")  # Path to dataset directory
    dataset_name: str = "fanta_single_data_KTO"  # Name of fine-tuning dataset
    run_root_dir: Path = Path("runs")  # Path to directory to store logs & checkpoints
    adapter_tmp_dir: Path = Path("adapter-tmp")  # Temporary directory for LoRA weights

    # Fine-tuning Parameters
    batch_size: int = 8  # Fine-tuning batch size
    max_steps: int = 20000  # Max number of fine-tuning steps
    save_steps: int = 500  # Interval for checkpoint saving
    learning_rate: float = 5e-4  # Fine-tuning learning rate
    grad_accumulation_steps: int = 1  # Gradient accumulation steps
    image_aug: bool = True  # Whether to train with image augmentations
    shuffle_buffer_size: int = 100_000  # Dataloader shuffle buffer size
    save_latest_checkpoint_only: bool = True  # Whether to save only the latest checkpoint

    # LoRA Arguments
    use_lora: bool = True  # Whether to use LoRA fine-tuning
    lora_rank: int = 32  # Rank of LoRA weight matrix
    lora_dropout: float = 0.0  # Dropout applied to LoRA weights
    use_quantization: bool = False  # Whether to 4-bit quantize VLA for LoRA fine-tuning

    # Tracking Parameters
    wandb_project: Optional[str] = None  # Name of W&B project to log to
    wandb_entity: Optional[str] = None   # Name of entity to log under
    run_id_note: Optional[str] = None    # Extra note for logging

    # KTO Loss Parameters
    beta: float = 1.0  # Beta parameter for KTO loss
    include_rewards: bool = True  # Default to True for KTO training

    grad_clip_norm: float = 1.0  # Maximum norm for gradient clipping


def get_batch_logps(logits, labels, label_pad_token_id=-100):
    """Compute log probabilities for tokens in the labels."""
    if logits.shape[:-1] != labels.shape:
        raise ValueError("Logits and labels must have same shape except for vocab dim")
    
    per_token_logps = torch.gather(
        logits.log_softmax(-1), 
        dim=2,
        index=labels.unsqueeze(2)
    ).squeeze(2)
    
    # Mask padded tokens
    loss_mask = labels != label_pad_token_id
    return (per_token_logps * loss_mask).sum(-1), loss_mask.sum(-1)

def kto_loss(
    policy_logps: torch.FloatTensor,
    reference_logps: torch.FloatTensor,
    rewards: torch.FloatTensor,
    beta: float = 0.1
) -> torch.FloatTensor:
    """Compute KTO loss using policy and reference log probs."""
    # Compute advantage (policy vs reference)
    advantage = policy_logps - reference_logps
    
    # KTO loss with continuous rewards
    losses = 1 - torch.sigmoid(beta * (advantage - (1 - rewards)))
    return losses.mean()

def save_checkpoint(vla, optimizer, step, cfg):
    """Save model checkpoint."""
    save_dir = cfg.run_root_dir
    
    if cfg.save_latest_checkpoint_only:
        # Overwrite latest checkpoint
        vla.module.save_pretrained(save_dir)
        print(f"Saved Model Checkpoint for Step {step} at: {save_dir}")
    else:
        # Save in new directory with step number
        checkpoint_dir = Path(str(save_dir) + f"--{step}_chkpt")
        os.makedirs(checkpoint_dir, exist_ok=True)
        vla.module.save_pretrained(checkpoint_dir)
        print(f"Saved Model Checkpoint for Step {step} at: {checkpoint_dir}")

def finetune(cfg: FinetuneConfig) -> None:
    try:
        # Initialize distributed setup with explicit device mapping
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        global_rank = int(os.environ.get("RANK", 0))
        
        logger = logging.getLogger(__name__)
        logger.info(f"{local_rank} - Starting setup")
        
        # Set device and seeds for reproducibility
        torch.manual_seed(42)
        device = torch.device(f'cuda:{local_rank}')
        
        # Print GPU memory status
        logger.info(f"{local_rank} - Initial GPU memory: {torch.cuda.memory_allocated(device) / 1024**2:.2f}MB")
        
        # Initialize process group with more error handling
        if not dist.is_initialized():
            try:
                logger.info(f"{local_rank} - Initializing process group")
                dist.init_process_group(
                    backend='nccl',
                    init_method='env://',
                    world_size=world_size,
                    rank=global_rank,
                    timeout=datetime.timedelta(minutes=60)
                )
                logger.info(f"{local_rank} - Process group initialized successfully")
            except Exception as e:
                logger.error(f"{local_rank} - Failed to initialize process group: {str(e)}")
                raise

        # Load models with memory tracking
        logger.info(f"{local_rank} - Starting model load")
        try:
            with torch.cuda.device(device):
                torch.cuda.empty_cache()
                logger.info(f"{local_rank} - Loading model to device {device}")
                
                vla = AutoModelForVision2Seq.from_pretrained(
                    cfg.vla_path,
                    torch_dtype=torch.bfloat16,
                    low_cpu_mem_usage=True,
                    trust_remote_code=True,
                    device_map={'': device}
                )
                logger.info(f"{local_rank} - Model loaded. GPU memory: {torch.cuda.memory_allocated(device) / 1024**2:.2f}MB")
                
                # Create reference model
                reference_vla = copy.deepcopy(vla)
                logger.info(f"{local_rank} - Reference model created. GPU memory: {torch.cuda.memory_allocated(device) / 1024**2:.2f}MB")
                
        except Exception as e:
            logger.error(f"{local_rank} - Failed to load models: {str(e)}")
            raise

        # Synchronize with logging
        logger.info(f"{local_rank} - Waiting at barrier after model load")
        dist.barrier()
        logger.info(f"{local_rank} - Passed barrier after model load")
        
        # Verify models across ranks before DDP
        param_tensor = torch.tensor([param_count], device=device)
        dist.all_reduce(param_tensor, op=dist.ReduceOp.MAX)
        if param_tensor.item() != param_count:
            raise ValueError(f"Rank {local_rank} has inconsistent parameters: {param_count} vs {param_tensor.item()}")
        
        logger.info(f"{local_rank} - Parameter verification complete")
        dist.barrier()
        
        # Initialize DDP with careful settings
        vla = DDP(
            vla,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,  # Changed to False for better performance
            broadcast_buffers=True,
            gradient_as_bucket_view=True,  # Memory optimization
            static_graph=True  # Performance optimization
        )
        
        reference_vla = DDP(
            reference_vla,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
            broadcast_buffers=True,
            static_graph=True
        )
        
        # Freeze reference model
        reference_vla.eval()
        for param in reference_vla.parameters():
            param.requires_grad = False
        
        dist.barrier()
        logger.info(f"{local_rank} - DDP initialization complete")
        
        # Create optimizer with gradient clipping
        optimizer = AdamW(
            [p for p in vla.parameters() if p.requires_grad],
            lr=cfg.learning_rate,
            eps=1e-8,  # Added for numerical stability
            weight_decay=0.01  # Added for stability
        )
        
        # Load dataset with error checking
        try:
            dataset = RLDSDataset(
                root_dir=cfg.data_root_dir,
                dataset_name=cfg.dataset_name,
                split="train",
                shuffle_buffer_size=cfg.shuffle_buffer_size
            )
            logger.info(f"{local_rank} - Dataset loaded successfully with {len(dataset)} samples")
        except Exception as e:
            logger.error(f"{local_rank} - Dataset loading failed: {str(e)}")
            raise

        # Training loop
        with tqdm.tqdm(total=cfg.max_steps, disable=local_rank != 0) as progress:
            vla.train()
            logger = logging.getLogger(__name__)
            logger.info(f"{local_rank} - Starting training initialization")

            # After loading models
            logger.info(f"{local_rank} - Models loaded successfully")

            # After dataset creation
            logger.info(f"{local_rank} - Dataset loaded, size: {len(dataset)}")

            # Inside training loop, before the batch loop
            batch_times = deque(maxlen=100)
            data_load_times = deque(maxlen=100)
            forward_times = deque(maxlen=100)

            for step in range(cfg.max_steps):
                try:
                    for batch in dataloader:
                        batch_start = time.time()
                        data_load_end = time.time()
                        data_load_times.append(data_load_end - batch_start)

                        # Move batch to device
                        batch = {k: v.to(device) if torch.is_tensor(v) else v 
                               for k, v in batch.items()}

                        optimizer.zero_grad()
                        
                        # Forward passes
                        forward_start = time.time()
                        with torch.autocast("cuda", dtype=torch.bfloat16):
                            policy_outputs = vla(**batch)
                            with torch.no_grad():
                                ref_outputs = reference_vla(**batch)
                        forward_end = time.time()
                        forward_times.append(forward_end - forward_start)

                        # Compute KTO loss
                        policy_logps, _ = get_batch_logps(
                            policy_outputs.logits,
                            batch["labels"]
                        )
                        ref_logps, _ = get_batch_logps(
                            ref_outputs.logits,
                            batch["labels"]
                        )
                        
                        loss = kto_loss(
                            policy_logps=policy_logps,
                            reference_logps=ref_logps,
                            rewards=batch["rewards"],
                            beta=cfg.beta
                        )

                        # Backward and optimize
                        loss.backward()
                        if cfg.grad_clip_norm > 0:
                            torch.nn.utils.clip_grad_norm_(
                                vla.parameters(),
                                cfg.grad_clip_norm
                            )
                        optimizer.step()

                        # Logging
                        if dist.get_rank() == 0:
                            progress.update(1)
                            progress.set_postfix(loss=loss.item())
                            
                            if use_wandb and step % 10 == 0:
                                wandb.log({
                                    "loss": loss.item(),
                                    "policy_logps": policy_logps.mean().item(),
                                    "ref_logps": ref_logps.mean().item(),
                                    "rewards": batch["rewards"].mean().item()
                                })

                        # Save checkpoint
                        if step > 0 and step % cfg.save_steps == 0 and dist.get_rank() == 0:
                            save_checkpoint(vla, optimizer, step, cfg)
                            dist.barrier()  # Synchronize after saving

                        # Add periodic timing logs
                        if dist.get_rank() == 0 and step % 10 == 0:
                            avg_batch_time = sum(batch_times) / len(batch_times) if batch_times else 0
                            avg_data_time = sum(data_load_times) / len(data_load_times) if data_load_times else 0
                            avg_forward_time = sum(forward_times) / len(forward_times) if forward_times else 0
                            
                            logger.info(
                                f"Step {step} timing - "
                                f"Batch: {avg_batch_time:.3f}s, "
                                f"Data loading: {avg_data_time:.3f}s, "
                                f"Forward pass: {avg_forward_time:.3f}s"
                            )

                        batch_end = time.time()
                        batch_times.append(batch_end - batch_start)

                except Exception as e:
                    logger.error(f"{local_rank} - Error in training loop: {str(e)}", exc_info=True)
                    raise e

    except Exception as e:
        logger.error(f"{local_rank} - Fatal error: {str(e)}", exc_info=True)
        # Cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if dist.is_initialized():
            dist.destroy_process_group()
        raise e

    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if dist.is_initialized():
            dist.destroy_process_group()
        if use_wandb and dist.get_rank() == 0:
            wandb.finish()


if __name__ == "__main__":
    config = FinetuneConfig(
        vla_path="openvla/openvla-7b",
        data_root_dir=Path("tensorflow_datasets/fanta_single_short_ft_dataset"),
        dataset_name="fanta_single_data_KTO",
        # Disable W&B for now
        wandb_project=None,
        wandb_entity=None
    )
    
    finetune(config)
