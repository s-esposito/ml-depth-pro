from PIL import Image
import depth_pro
import argparse
import os
import torch
import imageio
import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scene-path",
        help="path to the scene",
        type=str,
        default="/home/stefano/Codebase/DynSLAM/data/davis/car-turn",
    )
    args = parser.parse_args()
    print(args)
    
    # # images_paths
    # images_paths = glob.glob(
    #     os.path.join(args.scene_path, "rgba/rgba_*.png")
    # )
    #
    # images_paths
    images_paths = glob.glob(
        os.path.join(args.scene_path, "rgb/*.jpg")
    )
    images_paths.sort()
    print(images_paths)

    results_path = os.path.join(args.scene_path, "depth")
    os.system(f"mkdir -p {results_path}")
    
    # Load model and preprocessing transform
    model, transform = depth_pro.create_model_and_transforms(device="cuda:0")
    model.eval()
    
    pred_fxs = []
    pred_depths = []
    for image_path in tqdm(images_paths):

        # Load and preprocess an image.
        image, _, _ = depth_pro.load_rgb(image_path)
        f_px = torch.tensor(960.0)
        # print("image", image.shape)
        print("f_px", f_px)
        image = transform(image)
        
        # Run inference.
        prediction = model.infer(image, f_px=f_px)
        depth = prediction["depth"]  # Depth in [m].
        print("depth", depth.shape)
        focallength_px = prediction["focallength_px"]  # Focal length in pixels.
        print("focallength_px", focallength_px)
        pred_fxs.append(focallength_px.item())
        
        pred_depths.append(depth.cpu().numpy())
        
    # appy same focal lenght to all depths (average)
    avg_fx = np.array(pred_fxs).mean()
    print("avg_fx", avg_fx)
    
    corrected_depths = []
    for depth, f_px in zip(pred_depths, pred_fxs):
        # depth = depth * (avg_fx / f_px)
        corrected_depth = depth * (avg_fx / f_px)
        corrected_depths.append(corrected_depth)

    def convert_float_to_uint16(array, min_val, max_val):
        return np.round((array - min_val) / (max_val - min_val) * 65535).astype(np.uint16)

    def write_tiff(data: np.ndarray, filename: str):
        """Save data as as tif image (which natively supports float values)."""
        assert data.ndim == 3, data.shape
        assert data.shape[2] in [1, 3, 4], "Must be grayscale, RGB, or RGBA"

        img_as_bytes = imageio.imwrite("<bytes>", data, format="tiff")
        with open(filename, "wb") as f:
            f.write(img_as_bytes)
    
    # Save the results as .tiff
    # stack images along the first axis
    depth_frames = np.stack(corrected_depths, axis=0)
    print("depth_frames", depth_frames.shape)
    depth_min, depth_max = np.min(depth_frames), np.max(depth_frames)
    for i, depth in enumerate(depth_frames):
        i_str = format(i, "05d")
        # save an npy
        np.save(os.path.join(results_path, f"{i_str}.npy"), depth)
        # depth_unsqueezed = depth[..., np.newaxis]
        # repeat over the third axis
        # depth_unsqueezed = np.repeat(depth_unsqueezed, 3, axis=0)
        # Convert the depth map to uint16 format
        # depth_uint16 = convert_float_to_uint16(depth_unsqueezed, depth_min, depth_max)
        # print("depth_uint16", depth_uint16.shape)
        # write_tiff(depth_uint16, os.path.join(results_path, f"depth_{i_str}.tiff"))
        
if __name__ == "__main__":
    main()