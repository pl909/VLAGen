"""
Evaluate a model on ManiSkill2 environment.
"""

import os
import cv2
import re
import json
import numpy as np
from transforms3d.euler import quat2euler

from simpler_env.utils.env.env_builder import build_maniskill2_env, get_robot_control_mode
from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict
from simpler_env.utils.visualization import write_video
from simpler_env.evaluation.gpt_rate import rate_episodes_with_gpt4v

def run_maniskill2_eval_single_episode(
    model,
    ckpt_path,
    robot_name,
    env_name,
    scene_name,
    robot_init_x,
    robot_init_y,
    robot_init_quat,
    control_mode,
    obj_init_x=None,
    obj_init_y=None,
    obj_episode_id=None,
    additional_env_build_kwargs=None,
    rgb_overlay_path=None,
    obs_camera_name=None,
    control_freq=3,
    sim_freq=513,
    max_episode_steps=80,
    instruction=None,
    enable_raytracing=False,
    additional_env_save_tags=None,
    logging_dir="./results",
):

    if additional_env_build_kwargs is None:
        additional_env_build_kwargs = {}

    # Create environment
    kwargs = dict(
        obs_mode="rgbd",
        robot=robot_name,
        sim_freq=sim_freq,
        control_mode=control_mode,
        control_freq=control_freq,
        max_episode_steps=max_episode_steps,
        scene_name=scene_name,
        camera_cfgs={"add_segmentation": True},
        rgb_overlay_path=rgb_overlay_path,
    )
    if enable_raytracing:
        ray_tracing_dict = {"shader_dir": "rt"}
        ray_tracing_dict.update(additional_env_build_kwargs)
        # put raytracing dict keys before other keys for compatibility with existing result naming and metric calculation
        additional_env_build_kwargs = ray_tracing_dict
    env = build_maniskill2_env(
        env_name,
        **additional_env_build_kwargs,
        **kwargs,
    )

    # initialize environment
    env_reset_options = {
        "robot_init_options": {
            "init_xy": np.array([robot_init_x, robot_init_y]),
            "init_rot_quat": robot_init_quat,
        }
    }
    if obj_init_x is not None:
        assert obj_init_y is not None
        obj_variation_mode = "xy"
        env_reset_options["obj_init_options"] = {
            "init_xy": np.array([obj_init_x, obj_init_y]),
        }
    else:
        assert obj_episode_id is not None
        obj_variation_mode = "episode"
        env_reset_options["obj_init_options"] = {
            "episode_id": obj_episode_id,
        }
    obs, _ = env.reset(options=env_reset_options)
    # for long-horizon environments, we check if the current subtask is the final subtask
    is_final_subtask = env.unwrapped.is_final_subtask()

    # Obtain language instruction
    if instruction is not None:
        task_description = instruction
    else:
        # get default language instruction
        task_description = env.unwrapped.get_language_instruction()
    print(task_description)


    # Initialize logging
    image = get_image_from_maniskill2_obs_dict(env, obs, camera_name=obs_camera_name)
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
    images = [image]
    predicted_actions = []
    predicted_terminated, done, truncated = False, False, False

    # Initialize model
    model.reset(task_description)

    timestep = 0
    success = "failure"

    # Create debug image directory
    debug_image_dir = os.path.join(logging_dir, "debug_images")
    os.makedirs(debug_image_dir, exist_ok=True)

    # Step the environment
    while not (predicted_terminated or truncated):
        # step the model; "raw_action" is raw model action output; "action" is the processed action to be sent into maniskill env
        image = get_image_from_maniskill2_obs_dict(env, obs, camera_name=obs_camera_name)
        image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
        images.append(image)
        
        raw_action, action = model.step(image, task_description)
        predicted_actions.append(raw_action)
        predicted_terminated = bool(action["terminate_episode"][0] > 0)
        if predicted_terminated:
            if not is_final_subtask:
                # advance the environment to the next subtask
                predicted_terminated = False
                env.unwrapped.advance_to_next_subtask()

        # step the environment
        obs, reward, done, truncated, info = env.step(
            np.concatenate([action["world_vector"], action["rot_axangle"], action["gripper"]]),
        )
        
        success = "success" if done else "failure"
        new_task_description = env.unwrapped.get_language_instruction()
        if new_task_description != task_description:
            task_description = new_task_description
            print(task_description)
        is_final_subtask = env.unwrapped.is_final_subtask()

        print(timestep, info)

        # Save debug image with step number and object position
        debug_image = image.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(debug_image, f'Step: {timestep}, Obj: ({obj_init_x:.2f}, {obj_init_y:.2f})', 
                    (10, 30), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Convert to RGB before saving
        debug_image_rgb = cv2.cvtColor(debug_image, cv2.COLOR_BGR2RGB)
        
        # Save with informative filename
        debug_image_path = os.path.join(
            debug_image_dir, 
            f'debug_obj_{obj_init_x:.2f}_{obj_init_y:.2f}_step_{timestep:03d}.png'
        )
        cv2.imwrite(debug_image_path, debug_image_rgb)

    episode_stats = info.get("episode_stats", {})

    # save video
    env_save_name = env_name
    for k, v in additional_env_build_kwargs.items():
        env_save_name = env_save_name + f"_{k}_{v}"
    if additional_env_save_tags is not None:
        env_save_name = env_save_name + f"_{additional_env_save_tags}"
    ckpt_path_basename = ckpt_path if ckpt_path[-1] != "/" else ckpt_path[:-1]
    ckpt_path_basename = ckpt_path_basename.split("/")[-1]
    if obj_variation_mode == "xy":
        video_name = f"{success}_obj_{obj_init_x}_{obj_init_y}"
    elif obj_variation_mode == "episode":
        video_name = f"{success}_obj_episode_{obj_episode_id}"
    for k, v in episode_stats.items():
        video_name = video_name + f"_{k}_{v}"
    video_name = video_name + ".mp4"
    if rgb_overlay_path is not None:
        rgb_overlay_path_str = os.path.splitext(os.path.basename(rgb_overlay_path))[0]
    else:
        rgb_overlay_path_str = "None"
    r, p, y = quat2euler(robot_init_quat)
    video_path = f"{ckpt_path_basename}/{scene_name}/{control_mode}/{env_save_name}/rob_{robot_init_x}_{robot_init_y}_rot_{r:.3f}_{p:.3f}_{y:.3f}_rgb_overlay_{rgb_overlay_path_str}/{video_name}"
    video_path = os.path.join(logging_dir, video_path)
    write_video(video_path, images, fps=5)

    # save action trajectory
    action_path = video_path.replace(".mp4", ".png")
    action_root = os.path.dirname(action_path) + "/actions/"
    os.makedirs(action_root, exist_ok=True)
    action_path = action_root + os.path.basename(action_path)
    model.visualize_epoch(predicted_actions, images, save_path=action_path)

    return success == "success"




def run_maniskill2_eval_single_episode_generate_data(
    model,
    ckpt_path,
    robot_name,
    env_name,
    scene_name,
    robot_init_x,
    robot_init_y,
    robot_init_quat,
    control_mode,
    obj_init_x=None,
    obj_init_y=None,
    obj_episode_id=None,
    additional_env_build_kwargs=None,
    rgb_overlay_path=None,
    obs_camera_name=None,
    control_freq=3,
    sim_freq=513,
    max_episode_steps=80,
    instruction=None,
    enable_raytracing=False,
    additional_env_save_tags=None,
    data_dir=None,
    logging_dir="./results",
):

    if additional_env_build_kwargs is None:
        additional_env_build_kwargs = {}

    # Create environment
    kwargs = dict(
        obs_mode="rgbd",
        robot=robot_name,
        sim_freq=sim_freq,
        control_mode=control_mode,
        control_freq=control_freq,
        max_episode_steps=max_episode_steps,
        scene_name=scene_name,
        camera_cfgs={"add_segmentation": True},
        rgb_overlay_path=rgb_overlay_path,
    )
    if enable_raytracing:
        ray_tracing_dict = {"shader_dir": "rt"}
        ray_tracing_dict.update(additional_env_build_kwargs)
        # put raytracing dict keys before other keys for compatibility with existing result naming and metric calculation
        additional_env_build_kwargs = ray_tracing_dict
    env = build_maniskill2_env(
        env_name,
        **additional_env_build_kwargs,
        **kwargs,
    )

    # initialize environment
    env_reset_options = {
        "robot_init_options": {
            "init_xy": np.array([robot_init_x, robot_init_y]),
            "init_rot_quat": robot_init_quat,
        }
    }
    if obj_init_x is not None:
        assert obj_init_y is not None
        obj_variation_mode = "xy"
        env_reset_options["obj_init_options"] = {
            "init_xy": np.array([obj_init_x, obj_init_y]),
        }
    else:
        assert obj_episode_id is not None
        obj_variation_mode = "episode"
        env_reset_options["obj_init_options"] = {
            "episode_id": obj_episode_id,
        }
    obs, _ = env.reset(options=env_reset_options)
    # for long-horizon environments, we check if the current subtask is the final subtask
    is_final_subtask = env.unwrapped.is_final_subtask()
    # Initialize episode data collection
    episode_data = []
    
    # Obtain language instruction
    if instruction is not None:
        task_description = instruction
    else:
        # get default language instruction
        task_description = env.unwrapped.get_language_instruction()
    print(task_description)

    # Initialize logging
    image = get_image_from_maniskill2_obs_dict(env, obs, camera_name=obs_camera_name)
    resized_image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
    images = [resized_image]  # Store resized images for everything
    predicted_actions = []
    predicted_terminated, done, truncated = False, False, False

    # Initialize model
    model.reset(task_description)

    timestep = 0
    success = "failure"

    # Step the environment
    while not (predicted_terminated or truncated):
        # step the model
        raw_action, action = model.step(resized_image, task_description)  # Use resized image
        predicted_actions.append(raw_action)
        predicted_terminated = bool(action["terminate_episode"][0] > 0)
        if predicted_terminated:
            if not is_final_subtask:
                # advance the environment to the next subtask
                predicted_terminated = False
                env.unwrapped.advance_to_next_subtask()

        # Collect step data without reward
        step_data = {
            'image': resized_image,  # This will be 224x224x3
            'action': raw_action,
            'language_instruction': task_description,
        }
        episode_data.append(step_data)

        if predicted_terminated and not is_final_subtask:
            predicted_terminated = False
            env.unwrapped.advance_to_next_subtask()

        # step the environment
        obs, reward, done, truncated, info = env.step(
            np.concatenate([action["world_vector"], action["rot_axangle"], action["gripper"]]),
        )
        
        success = "success" if done else "failure"
        new_task_description = env.unwrapped.get_language_instruction()
        if new_task_description != task_description:
            task_description = new_task_description
            print(task_description)
        is_final_subtask = env.unwrapped.is_final_subtask()

        print(timestep, info)

        image = get_image_from_maniskill2_obs_dict(env, obs, camera_name=obs_camera_name)
        resized_image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
        images.append(resized_image)  # Store resized image for video too
        timestep += 1

    episode_stats = info.get("episode_stats", {})
    # Save episode data
    env_save_name = env_name
    for k, v in additional_env_build_kwargs.items():
        env_save_name = env_save_name + f"_{k}_{v}"
    if additional_env_save_tags is not None:
        env_save_name = env_save_name + f"_{additional_env_save_tags}"
        
    ckpt_path_basename = ckpt_path if ckpt_path[-1] != "/" else ckpt_path[:-1]
    ckpt_path_basename = ckpt_path_basename.split("/")[-1]
    
    r, p, y = quat2euler(robot_init_quat)
    if obj_init_x is not None and obj_init_y is not None:
        episode_path = f"{ckpt_path_basename}/{scene_name}/{control_mode}/{env_save_name}/rob_{robot_init_x}_{robot_init_y}_rot_{r:.3f}_{p:.3f}_{y:.3f}_obj_{obj_init_x}_{obj_init_y}_episode_{obj_episode_id}.npy"
    else:
        episode_path = f"{ckpt_path_basename}/{scene_name}/{control_mode}/{env_save_name}/rob_{robot_init_x}_{robot_init_y}_rot_{r:.3f}_{p:.3f}_{y:.3f}_episode_{obj_episode_id}.npy"
    episode_path = os.path.join(data_dir, "data", episode_path)
    
    os.makedirs(os.path.dirname(episode_path), exist_ok=True)
    #np.save(episode_path, episode_data)

    # Save final image with episode ID label
    final_image = images[-1].copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(final_image, f'Episode {obj_episode_id}', 
                (10, 30), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
    # Convert to RGB before saving
    final_image_rgb = cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB)

    # Construct the path for the final image
    if obj_init_x is not None and obj_init_y is not None:
        final_image_path = os.path.join(
            os.path.dirname(episode_path),
            f'final_image_obj_{obj_init_x}_{obj_init_y}_episode_{obj_episode_id}.png'
        )
    else:
        final_image_path = os.path.join(
            os.path.dirname(episode_path),
            f'final_image_episode_{obj_episode_id}.png'
        )

    cv2.imwrite(final_image_path, final_image_rgb)


    # save video
    env_save_name = env_name
    for k, v in additional_env_build_kwargs.items():
        env_save_name = env_save_name + f"_{k}_{v}"
    if additional_env_save_tags is not None:
        env_save_name = env_save_name + f"_{additional_env_save_tags}"
    ckpt_path_basename = ckpt_path if ckpt_path[-1] != "/" else ckpt_path[:-1]
    ckpt_path_basename = ckpt_path_basename.split("/")[-1]
    if obj_variation_mode == "xy":
        video_name = f"{success}_obj_{obj_init_x}_{obj_init_y}"
    elif obj_variation_mode == "episode":
        video_name = f"{success}_obj_episode_{obj_episode_id}"
    for k, v in episode_stats.items():
        video_name = video_name + f"_{k}_{v}"
    video_name = video_name + ".mp4"
    if rgb_overlay_path is not None:
        rgb_overlay_path_str = os.path.splitext(os.path.basename(rgb_overlay_path))[0]
    else:
        rgb_overlay_path_str = "None"
    r, p, y = quat2euler(robot_init_quat)
    if obj_init_x is not None and obj_init_y is not None:
        video_path = f"{ckpt_path_basename}/{scene_name}/{control_mode}/{env_save_name}/rob_{robot_init_x}_{robot_init_y}_rot_{r:.3f}_{p:.3f}_{y:.3f}_obj_{obj_init_x}_{obj_init_y}_episode_{obj_episode_id}_rgb_overlay_{rgb_overlay_path_str}/{video_name}"
    else:
        video_path = f"{ckpt_path_basename}/{scene_name}/{control_mode}/{env_save_name}/rob_{robot_init_x}_{robot_init_y}_rot_{r:.3f}_{p:.3f}_{y:.3f}_episode_{obj_episode_id}_rgb_overlay_{rgb_overlay_path_str}/{video_name}"
    video_path = os.path.join(data_dir, "data", video_path)
    write_video(video_path, images, fps=5)

    # save action trajectory
    action_path = video_path.replace(".mp4", ".png")
    action_root = os.path.dirname(action_path) + "/actions/"
    os.makedirs(action_root, exist_ok=True)
    action_path = action_root + os.path.basename(action_path)
    model.visualize_epoch(predicted_actions, images, save_path=action_path)

    return success == "success", final_image, episode_path, episode_data


def maniskill2_evaluator(model, args):
    control_mode = get_robot_control_mode(args.robot, args.policy_model)
    success_arr = []

    # run inference
    for robot_init_x in args.robot_init_xs:
        for robot_init_y in args.robot_init_ys:
            for robot_init_quat in args.robot_init_quats:
                kwargs = dict(
                    model=model,
                    ckpt_path=args.ckpt_path,
                    robot_name=args.robot,
                    env_name=args.env_name,
                    scene_name=args.scene_name,
                    robot_init_x=robot_init_x,
                    robot_init_y=robot_init_y,
                    robot_init_quat=robot_init_quat,
                    control_mode=control_mode,
                    additional_env_build_kwargs=args.additional_env_build_kwargs,
                    rgb_overlay_path=args.rgb_overlay_path,
                    control_freq=args.control_freq,
                    sim_freq=args.sim_freq,
                    max_episode_steps=args.max_episode_steps,
                    enable_raytracing=args.enable_raytracing,
                    additional_env_save_tags=args.additional_env_save_tags,
                    obs_camera_name=args.obs_camera_name,
                    logging_dir=args.logging_dir,
                )
                if args.obj_variation_mode == "xy":
                    for obj_init_x in args.obj_init_xs:
                        for obj_init_y in args.obj_init_ys:
                            print(obj_init_x)
                            print(obj_init_y)
                            success_arr.append(
                                run_maniskill2_eval_single_episode(
                                    obj_init_x=obj_init_x,
                                    obj_init_y=obj_init_y,
                                    **kwargs,
                                )
                            )
                elif args.obj_variation_mode == "episode":
                    for obj_episode_id in range(args.obj_episode_range[0], args.obj_episode_range[1]):
                        success_arr.append(run_maniskill2_eval_single_episode(obj_episode_id=obj_episode_id, **kwargs))
                else:
                    raise NotImplementedError()

    return success_arr

# new method for OpenVLA KTO training

def maniskill2_dataset_generator(model, args):
    control_mode = get_robot_control_mode(args.robot, args.policy_model)
    success_arr = []

    # run inference
    for robot_init_x in args.robot_init_xs:
        for robot_init_y in args.robot_init_ys:
            for robot_init_quat in args.robot_init_quats:
                kwargs = dict(
                    model=model,
                    ckpt_path=args.ckpt_path,
                    robot_name=args.robot,
                    env_name=args.env_name,
                    scene_name=args.scene_name,
                    robot_init_x=robot_init_x,
                    robot_init_y=robot_init_y,
                    robot_init_quat=robot_init_quat,
                    control_mode=control_mode,
                    additional_env_build_kwargs=args.additional_env_build_kwargs,
                    rgb_overlay_path=args.rgb_overlay_path,
                    control_freq=args.control_freq,
                    sim_freq=args.sim_freq,
                    max_episode_steps=args.max_episode_steps,
                    enable_raytracing=args.enable_raytracing,
                    additional_env_save_tags=args.additional_env_save_tags,
                    obs_camera_name=args.obs_camera_name,
                    logging_dir=args.logging_dir,
                )
                if args.obj_variation_mode == "xy":
                    for obj_init_x in args.obj_init_xs:
                        for obj_init_y in args.obj_init_ys:
                            episode_datas = []
                            episode_paths = []
                            final_images = []

                            # First collect all episodes
                            for i in range(8):
                                success, final_image, episode_path, episode_data = run_maniskill2_eval_single_episode_generate_data(
                                    obj_init_x=obj_init_x,
                                    obj_init_y=obj_init_y,
                                    obj_episode_id=i,
                                    data_dir=args.data_dir,
                                    **kwargs,
                                )
                                success_arr.append(success)
                                episode_datas.append(episode_data)
                                episode_paths.append(episode_path)

                                # Add episode number to final image and collect it
                                labeled_image = cv2.cvtColor(final_image.copy(), cv2.COLOR_BGR2RGB)
                                font = cv2.FONT_HERSHEY_SIMPLEX
                                cv2.putText(labeled_image, f'Episode {i}', 
                                            (10, 30), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
                                final_images.append(labeled_image)
                            
                            # Get GPT-4V rating using the images directly from memory
                            task_description = episode_datas[0][0]['language_instruction']
                            gpt_rating = rate_episodes_with_gpt4v(final_images, task_description)
                            print(f"\nGPT-4V Rating for position ({obj_init_x}, {obj_init_y}):")
                            print(gpt_rating)

                            # Extract the JSON object from gpt_rating with improved regex
                            json_match = re.search(r'```json\s*(\{[\s\S]*?\})\s*```', gpt_rating)
                            
                            if json_match:
                                json_str = json_match.group(1)
                                try:
                                    ratings = json.loads(json_str)
                                    for episode_idx in range(8):
                                        reward = float(ratings.get(str(episode_idx), 0)) / 10.0
                                        for step_idx in range(len(episode_datas[episode_idx])):
                                            episode_datas[episode_idx][step_idx]['reward'] = reward
                                    print(f"Successfully parsed JSON ratings: {ratings}")
                                except json.JSONDecodeError as e:
                                    print(f"Error parsing JSON: {e}, defaulting rewards to 0.0")
                                    for episode_idx in range(8):
                                        for step_idx in range(len(episode_datas[episode_idx])):
                                            episode_datas[episode_idx][step_idx]['reward'] = 0.0
                            else:
                                print("No valid JSON found in the following GPT rating:")
                                print(gpt_rating)
                                print("Defaulting rewards to 0.0")
                                for episode_idx in range(8):
                                    for step_idx in range(len(episode_datas[episode_idx])):
                                        episode_datas[episode_idx][step_idx]['reward'] = 0.0

                            # Print final rewards
                            for episode_idx in range(8):
                                print(f"Final reward for episode {episode_idx}: {episode_datas[episode_idx][0]['reward']}")

                            # Save episode after rewards are added
                            for episode_idx in range(8):
                                os.makedirs(os.path.dirname(episode_paths[episode_idx]), exist_ok=True)
                                np.save(episode_paths[episode_idx], episode_datas[episode_idx])
                elif args.obj_variation_mode == "episode":
                    for obj_episode_id in range(args.obj_episode_range[0], args.obj_episode_range[1]):
                        success_arr.append(run_maniskill2_eval_single_episode(obj_episode_id=obj_episode_id, **kwargs))
                else:
                    raise NotImplementedError()

    return success_arr








