from typing import Optional, Sequence
import os
import matplotlib.pyplot as plt
import numpy as np
from transforms3d.euler import euler2axangle
from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image
import torch
import cv2 as cv
import gc
from transformers import AutoConfig, AutoImageProcessor



class OpenVLAInference:
    def __init__(
    self,
    saved_model_path: str = "/home/pl217/fantadistractornew",
    unnorm_key: Optional[str] = None,
    policy_setup: str = "google_robot",
    horizon: int = 1,
    pred_action_horizon: int = 1,
    exec_horizon: int = 1,
    image_size: list[int] = [224, 224],
    action_scale: float = 1.0,
) -> None:
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
            
        if policy_setup == "widowx_bridge":
            self.unnorm_key = "bridge_orig" 
            self.sticky_gripper_num_repeat = 1
        elif policy_setup == "google_robot":
            self.unnorm_key = "fractal20220817_data" 
            self.sticky_gripper_num_repeat = 15
        else:
            raise NotImplementedError(
                f"Policy setup {policy_setup} not supported for octo models."
            )
        self.policy_setup = policy_setup
        # Load model and processor
        config = AutoConfig.from_pretrained(saved_model_path, trust_remote_code=True)
        self.processor = AutoProcessor.from_pretrained(
            saved_model_path,
            trust_remote_code=True,
            config=config
        )
        self.vla = AutoModelForVision2Seq.from_pretrained(
            saved_model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            config=config
        ).cuda()
        self.image_size = image_size
        self.action_scale = action_scale
        self.horizon = horizon
        self.pred_action_horizon = pred_action_horizon
        self.exec_horizon = exec_horizon

        self.sticky_action_is_on = False
        self.gripper_action_repeat = 0
        self.sticky_gripper_action = 0.0
        self.previous_gripper_action = None

        self.task = None
        self.task_description = None
        self.num_image_history = 0

        self.non_movement_counter = 0
        self.NON_MOVEMENT_THRESHOLD = 5  # Number of cycles before forcing movement
        self.MOVEMENT_THRESHOLD = 1e-6

    def reset(self, task_description: str) -> None:
        self.task_description = task_description
        self.num_image_history = 0

        self.sticky_action_is_on = False
        self.gripper_action_repeat = 0
        self.sticky_gripper_action = 0.0
        self.previous_gripper_action = None

    def step(
        self, image: np.ndarray, task_description: Optional[str] = None, *args, **kwargs
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """
        Input:
            image: np.ndarray of shape (H, W, 3), uint8
            task_description: Optional[str], task description
        Output:
            raw_action: dict
            action: dict
        """
        if task_description is not None:
            if task_description != self.task_description:
                self.reset(task_description)

        assert image.dtype == np.uint8
        image = self._resize_image(image)
        image = Image.fromarray(image)
        
        # Format prompt like in test_openvla.py
        prompt = task_description

        # Process inputs and get prediction (similar to test_openvla.py)
        inputs = self.processor(prompt, image).to("cuda:0", dtype=torch.bfloat16)
        predicted_action = self.vla.predict_action(**inputs, unnorm_key="fractal20220817_data")
        
        # Convert to numpy if not already
        predicted_action = np.array(predicted_action)
        
        print("Original action:", predicted_action)  # Print original action
        
        # Create raw_action dict
        raw_action = {
            "world_vector": predicted_action[:3],
            "rotation_delta": predicted_action[3:6],
            "open_gripper": predicted_action[6:7],
        }

        # Create action dict
        action = {}
        action["world_vector"] = raw_action["world_vector"]
        
        # Convert rotation
        action_rotation_delta = np.asarray(raw_action["rotation_delta"], dtype=np.float64)
        roll, pitch, yaw = action_rotation_delta
        action_rotation_ax, action_rotation_angle = euler2axangle(roll, pitch, yaw)
        action["rot_axangle"] = action_rotation_ax * action_rotation_angle

        # Gripper action
        action["gripper"] = raw_action["open_gripper"]
        action["terminate_episode"] = np.array([0.0])

        return raw_action, action

    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        image = cv.resize(image, tuple(self.image_size), interpolation=cv.INTER_AREA)
        return image

    def visualize_epoch(
        self, predicted_raw_actions: Sequence[np.ndarray], images: Sequence[np.ndarray], save_path: str
    ) -> None:
        images = [self._resize_image(image) for image in images]
        ACTION_DIM_LABELS = ["x", "y", "z", "roll", "pitch", "yaw", "grasp"]

        img_strip = np.concatenate(np.array(images[::3]), axis=1)

        # set up plt figure
        figure_layout = [["image"] * len(ACTION_DIM_LABELS), ACTION_DIM_LABELS]
        plt.rcParams.update({"font.size": 12})
        fig, axs = plt.subplot_mosaic(figure_layout)
        fig.set_size_inches([45, 10])

        # plot actions
        pred_actions = np.array(
            [
                np.concatenate([a["world_vector"], a["rotation_delta"], a["open_gripper"]], axis=-1)
                for a in predicted_raw_actions
            ]
        )
        for action_dim, action_label in enumerate(ACTION_DIM_LABELS):
            # actions have batch, horizon, dim, in this example we just take the first action for simplicity
            axs[action_label].plot(pred_actions[:, action_dim], label="predicted action")
            axs[action_label].set_title(action_label)
            axs[action_label].set_xlabel("Time in one episode")

        axs["image"].imshow(img_strip)
        axs["image"].set_xlabel("Time in one episode (subsampled)")
        plt.legend()
        plt.savefig(save_path)
