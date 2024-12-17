import os
import base64
import cv2
import openai

def encode_image_from_array(image_array):
    """Convert numpy array to base64 string."""
    success, buffer = cv2.imencode('.jpg', image_array)
    if not success:
        raise ValueError("Could not encode image")
    return base64.b64encode(buffer).decode('utf-8')

def rate_episodes_with_gpt4v(final_images, task_description):
    """
    Rate each episode's final state based on distance ratio to goal.
    
    Args:
        final_images: List of numpy arrays containing the final images with episode labels
        task_description: Description of the task that was supposed to be completed
    
    Returns:
        ratings: JSON with each episode rated from 0 to 10
    """
    client = openai.OpenAI()
    openai.api_key = os.getenv("OPENAI_API_KEY")
    
    message_content = [
        {
            "type": "text",
            "text": (
                f"You are looking at 8 final states of a robot attempting this task: {task_description}\n"
                f"Each image is labeled with its episode number (0-7).\n\n"
                f"Please analyze each image and rate it from 0-10 based on these criteria:\n"
                f"- The robot arm starts from the right edge of the screen\n"
                f"- The goal is to grip the object mentioned in the task description\n"
                f"- IMPORTANT RATING RULES:\n"
                f"  - If all robot grippers stayed at the right edge: Give all episodes a 0\n"
                f"  - Otherwise: Give the best attempt a 10, and rate others proportionally based on:\n"
                f"    - How close they got to the target relative to the best attempt\n"
                f"    - 0: Robot arm hasn't moved from starting position on right side\n"
                f"    - 5: Robot moved about half as far as the best attempt\n"
                f"    - 10: Best attempt (moved closest to / reached the target)\n\n"
                f"First, describe what you see in each episode (position of robot arm relative to target).\n"
                f"Then, provide your ratings as a JSON with integer values 0-10 for each episode.\n"
                f"Format: {{'0': rating, '1': rating, ..., '7': rating}}"
            )
        }
    ]
    
    # Add all images to the message directly from numpy arrays
    for image in final_images:
        base64_image = encode_image_from_array(image)
        message_content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}",
                "detail": "low"
            }
        })
    
    response = client.chat.completions.create(
        model="gpt-4o",  # Use appropriate model
        messages=[{
            "role": "user",
            "content": message_content
        }],
        max_tokens=1000  # Increased to allow for reasoning
    )
    
    return response.choices[0].message.content
