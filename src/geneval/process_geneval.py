import os
import shutil
import argparse

def process_images(parent_dir: str, steps: list[int]):
    """
    Process final_img and opt_history directories to extract optimized images
    for specified steps.

    Args:
        parent_dir (str): Parent directory containing final_img and opt_history folders
        steps (List[int]): List of optimization steps to extract
    """
    final_img_dir = os.path.join(parent_dir, "final_img")
    opt_history_dir = os.path.join(parent_dir, "opt_history")

    for step in steps:
        new_final_img_dir = os.path.join(parent_dir, f"final_img_{step:02d}")
        os.makedirs(new_final_img_dir, exist_ok=True)

        for example_dir in sorted(os.listdir(final_img_dir)):
            final_example_path = os.path.join(final_img_dir, example_dir)
            if not os.path.isdir(final_example_path):
                continue

            # Create new output path for this example
            new_example_path = os.path.join(new_final_img_dir, example_dir)
            os.makedirs(new_example_path, exist_ok=True)

            # Copy metadata.jsonl if it exists
            metadata_file = os.path.join(final_example_path, "metadata.jsonl")
            if os.path.exists(metadata_file):
                shutil.copy(metadata_file, os.path.join(new_example_path, "metadata.jsonl"))

            # Create samples subfolder
            new_samples_path = os.path.join(new_example_path, "samples")
            os.makedirs(new_samples_path, exist_ok=True)

            # Determine optimization folder name (last 4 characters)
            opt_dir_name = example_dir[-4:]
            opt_example_path = os.path.join(opt_history_dir, opt_dir_name)

            if not os.path.exists(opt_example_path):
                # No optimization history → use original image
                origin_img = os.path.join(final_example_path, "samples", "0000.png")
                if os.path.exists(origin_img):
                    shutil.copy(origin_img, os.path.join(new_samples_path, "0000.png"))
            else:
                # Find all optimized images sorted by step
                all_opt_imgs = sorted([
                    f for f in os.listdir(opt_example_path)
                    if f.startswith("optimized_image_") and f.endswith(".png")
                ], key=lambda x: int(x.split("_")[-1].split(".")[0]))

                total_optimized = len(all_opt_imgs)
                selected_index = min(step - 1, total_optimized - 1)
                selected_img_name = f"optimized_image_{selected_index}.png"
                selected_img_path = os.path.join(opt_example_path, selected_img_name)

                if os.path.exists(selected_img_path):
                    shutil.copy(selected_img_path, os.path.join(new_samples_path, "0000.png"))
                else:
                    print(f"[WARNING] {selected_img_path} does not exist. Skipped.")

    print("All steps processed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract optimized images for specific steps")
    parser.add_argument(
        "--parent_dir", type=str, required=True,
        help="Parent directory containing final_img/ and opt_history/"
    )
    parser.add_argument(
        "--steps", type=int, nargs="+", required=True,
        help="List of steps to extract, e.g., --steps 5 10 15 20"
    )

    args = parser.parse_args()
    process_images(args.parent_dir, args.steps)
