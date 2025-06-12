import os
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from sklearn.decomposition import PCA
import argparse

def load_mask(mask_path):
    # Loads a mask png (H, W, 4) and returns a binary mask (N, H, W) for each object
    mask_img = Image.open(mask_path).convert("RGBA")
    mask_np = np.array(mask_img)
    # The mask is RGBA, but each object is a different color (including alpha=255)
    # We'll treat each unique color (excluding fully transparent) as a separate object
    mask_flat = mask_np.reshape(-1, 4)
    unique_colors = np.unique(mask_flat, axis=0)
    # Exclude fully transparent
    unique_colors = [tuple(c) for c in unique_colors if c[3] > 0]
    masks = []
    for color in unique_colors:
        obj_mask = np.all(mask_np == color, axis=-1)
        if obj_mask.sum() > 0:
            masks.append(obj_mask)
    if len(masks) == 0:
        return np.zeros((0, mask_np.shape[0], mask_np.shape[1]), dtype=bool)
    return np.stack(masks, axis=0)

def get_clip_model():
    import maskclip_onnx
    model, preprocess = maskclip_onnx.clip.load(
        "ViT-L/14@336px",
        download_root=os.getenv('TORCH_HOME', os.path.join(os.path.expanduser('~'), '.cache', 'torch'))
    )
    model.eval()
    return model, preprocess

def compute_object_clip_embeddings(image_path, mask_path, output_dir, device="cuda"):
    # Load image
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)
    H, W = image_np.shape[:2]

    # Load masks
    masks = load_mask(mask_path)  # (N_obj, H, W)
    if masks.shape[0] == 0:
        print(f"No objects found in mask: {mask_path}")
        return

    # Try to load bounding boxes
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    bbox_path = os.path.join(output_dir, f"{base_name}_bboxes.npy")
    if not os.path.exists(bbox_path):
        bbox_path = mask_path.replace('_mask.png', '_bboxes.npy')
    bboxes = None
    if os.path.exists(bbox_path):
        bboxes = np.load(bbox_path)
        if bboxes.shape[0] != masks.shape[0]:
            print(f"Warning: Number of bboxes ({bboxes.shape[0]}) does not match number of masks ({masks.shape[0]}) for {image_path}. Falling back to mask-based bbox.")
            bboxes = None

    # Load CLIP model
    model, preprocess = get_clip_model()
    model = model.to(device)

    # Preprocessing for CLIP
    norm = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    transform = T.Compose([
        T.Resize((336, 336)),
        T.ToTensor(),
        norm
    ])

    for obj_idx, obj_mask in enumerate(masks):
        # Use bbox from file if available, otherwise compute from mask
        if bboxes is not None:
            x0, y0, x1, y1 = bboxes[obj_idx]
        else:
            ys, xs = np.where(obj_mask)
            if len(xs) == 0 or len(ys) == 0:
                continue
            x0, x1 = xs.min(), xs.max()
            y0, y1 = ys.min(), ys.max()
        # Crop the image and mask to the bounding box
        crop_img = image.crop((x0, y0, x1+1, y1+1))
        crop_mask = obj_mask[y0:y1+1, x0:x1+1]
        # Apply mask to the crop (set background to 0)
        crop_img_np = np.array(crop_img)
        crop_img_np[~crop_mask] = 0
        crop_img_masked = Image.fromarray(crop_img_np)

        # Preprocess and get CLIP embedding
        input_tensor = transform(crop_img_masked).unsqueeze(0).to(device)
        with torch.no_grad():
            feat = model.encode_image(input_tensor)
            feat = feat / feat.norm(dim=-1, keepdim=True)
        feat_np = feat.cpu().numpy()[0]  # (C,)

        # Create a pixel-wise embedding map for the object (all pixels in the mask get the same embedding)
        obj_feat_map = np.zeros((feat_np.shape[0], H, W), dtype=np.float32)
        obj_feat_map[:, obj_mask] = feat_np[:, None]

        # Save the pixel-wise embedding as .npy
        npy_save_path = os.path.join(output_dir, f"{base_name}_obj{obj_idx}_clip_feat.npy")
        np.save(npy_save_path, obj_feat_map)

        # Visualize the embedding using PCA
        # Only use the masked region for PCA
        masked_pixels = obj_feat_map[:, obj_mask].T  # (N_pixels, C)
        if masked_pixels.shape[0] < 3:
            continue  # Not enough pixels for PCA
        pca = PCA(n_components=3)
        pca_result = pca.fit_transform(masked_pixels)
        # Normalize to 0-255
        pca_result = (pca_result - pca_result.min()) / (pca_result.max() - pca_result.min())
        pca_result = (pca_result * 255).astype(np.uint8)
        # Create an RGB image for the object
        pca_img = np.zeros((H, W, 3), dtype=np.uint8)
        pca_img[obj_mask] = pca_result
        # Save PCA visualization
        pca_save_path = os.path.join(output_dir, f"{base_name}_obj{obj_idx}_clip_pca.png")
        Image.fromarray(pca_img).save(pca_save_path)

def process_all_masks(image_dir, mask_dir, output_dir, device="cuda"):
    os.makedirs(output_dir, exist_ok=True)
    
    # Try to find images in 'images' or 'color' subdirectories
    possible_image_dirs = [
        os.path.join(image_dir, 'images'),
        os.path.join(image_dir, 'color'),
        image_dir  # fallback to the provided directory
    ]
    
    image_dir = None
    for dir_path in possible_image_dirs:
        if os.path.exists(dir_path):
            image_dir = dir_path
            print(f"Found images in: {dir_path}")
            break
    
    if image_dir is None:
        raise ValueError(f"No image directory found in {image_dir}. Tried: {possible_image_dirs}")

    for fname in os.listdir(mask_dir):
        if fname.endswith("_mask.png"):
            mask_path = os.path.join(mask_dir, fname)
            base_name = fname.replace("_mask.png", "")
            # Try to find the corresponding image
            image_found = False
            for ext in [".png", ".jpg", ".jpeg"]:
                image_path = os.path.join(image_dir, base_name + ext)
                if os.path.exists(image_path):
                    image_found = True
                    break
            if not image_found:
                print(f"Image for mask {fname} not found in {image_dir}.")
                continue
            compute_object_clip_embeddings(image_path, mask_path, output_dir, device=device)

def main():
    parser = argparse.ArgumentParser(description="Compute object CLIP embeddings and PCA visualizations from masks.")
    parser.add_argument('--image_dir', type=str, required=True, help='Directory containing input images')
    parser.add_argument('--mask_dir', type=str, required=True, help='Directory containing mask images (e.g., *_mask.png)')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save output features and visualizations')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda or cpu)')
    args = parser.parse_args()

    process_all_masks(
        image_dir=args.image_dir,
        mask_dir=args.mask_dir,
        output_dir=args.output_dir,
        device=args.device
    )

if __name__ == "__main__":
    main()

# Example usage:
# process_all_masks(
#     image_dir="path/to/images",
#     mask_dir="path/to/sam_clip_features",
#     output_dir="path/to/clip_object_features",
#     device="cuda"
# )
