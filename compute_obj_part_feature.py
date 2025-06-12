import os
from utils.general_utils import pytorch_gc
from argparse import ArgumentParser
from PIL import Image
import numpy as np
import torch
from tqdm import tqdm, trange
import cv2
from typing import Any, Dict, Generator, List, Tuple
import matplotlib.pyplot as plt
import numpy as np

import torch.nn as nn
import torchvision.transforms as T
import torch.nn.functional as F
import matplotlib.pyplot as plt
import maskclip_onnx
from sklearn.decomposition import PCA

def resize_image(img, longest_edge):
    # resize to have the longest edge equal to longest_edge
    width, height = img.size
    if width > height:
        ratio = longest_edge / width
    else:
        ratio = longest_edge / height
    new_width = int(width * ratio)
    new_height = int(height * ratio)
    return img.resize((new_width, new_height), Image.BILINEAR)

def interpolate_to_patch_size(img_bchw, patch_size):
    # Interpolate the image so that H and W are multiples of the patch size
    _, _, H, W = img_bchw.shape
    target_H = H // patch_size * patch_size
    target_W = W // patch_size * patch_size
    img_bchw = torch.nn.functional.interpolate(img_bchw, size=(target_H, target_W))
    return img_bchw, target_H, target_W

def is_valid_image(filename):
    ext_test_flag = any(filename.lower().endswith(extension) for extension in ['.png', '.jpg', '.jpeg'])
    is_file_flag = os.path.isfile(filename)
    return ext_test_flag and is_file_flag
    
def show_anns(anns):
    if len(anns) == 0:
        return
    img = np.ones((anns.shape[1], anns.shape[2], 4))
    img[:,:,3] = 0
    for ann in range(anns.shape[0]):
        m = anns[ann].bool()
        m=m.cpu().numpy()
        color_mask = np.concatenate([np.random.random(3), [1]])
        img[m] = color_mask
    return img

def batch_iterator(batch_size: int, *args) -> Generator[List[Any], None, None]:
    assert len(args) > 0 and all(
        len(a) == len(args[0]) for a in args
    ), "Batched iteration must have inputs of all the same size."
    n_batches = len(args[0]) // batch_size + int(len(args[0]) % batch_size != 0)
    for b in range(n_batches):
        yield [arg[b * batch_size : (b + 1) * batch_size] for arg in args]


# maskclip_onnx is a library that allows us to use the CLIP model to extract features from images
# it is a wrapper around the CLIP model that allows us to use it in a more convenient way
# it is a PyTorch module that can be used in a similar way to a regular PyTorch model
class MaskCLIPFeaturizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.model, self.preprocess = maskclip_onnx.clip.load(
            "ViT-L/14@336px",
            download_root=os.getenv('TORCH_HOME', os.path.join(os.path.expanduser('~'), '.cache', 'torch'))
        )
        self.model.eval()
        self.patch_size = self.model.visual.patch_size

    def forward(self, img):
        b, _, input_size_h, input_size_w = img.shape
        patch_h = input_size_h // self.patch_size
        patch_w = input_size_w // self.patch_size
        features = self.model.get_patch_encodings(img).to(torch.float32)
        return features.reshape(b, patch_h, patch_w, -1).permute(0, 3, 1, 2)

def visualize_aggregated_feat_map(aggregated_feat_map, save_path, original_size=None):
    """Visualize aggregated feature map using PCA."""
    # Move tensor to CPU and convert to numpy
    if torch.is_tensor(aggregated_feat_map):
        aggregated_feat_map = aggregated_feat_map.cpu().numpy()
    
    # Reshape the feature map to (num_features, height * width)
    num_features, height, width = aggregated_feat_map.shape
    reshaped_feat_map = aggregated_feat_map.reshape(num_features, -1).T

    # Apply PCA to reduce to 3 components
    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(reshaped_feat_map)

    # Reshape PCA result back to (height, width, 3)
    pca_image = pca_result.reshape(height, width, 3)

    # Normalize the PCA image to 0-255 and convert to uint8
    pca_image = (pca_image - pca_image.min()) / (pca_image.max() - pca_image.min())
    pca_image = (pca_image * 255).astype(np.uint8)

    # If original_size is provided, upsample to that size
    img = Image.fromarray(pca_image)
    if original_size is not None:
        img = img.resize(original_size, Image.NEAREST)
    img.save(save_path)

def setup_transforms(args):
    """Setup all necessary transforms for different models."""
    norm = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    
    part_transform = T.Compose([
        T.Resize((args.part_resolution, args.part_resolution)),
        T.ToTensor(),
        norm
    ])

    raw_transform = T.Compose([
        T.ToTensor(),
        norm
    ])

    dino_transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.5], std=[0.5]),
    ])
    
    return {
        'part_transform': part_transform,
        'raw_transform': raw_transform,
        'dino_transform': dino_transform
    }

# See the following for more details:
# https://github.com/RogerQi/MobileSAMV2/blob/main/hubconf.py
# This gives us the models used for mobilesamev2, objawaremodel, and predictor
# Sam Predictor: https://github.com/RogerQi/MobileSAMV2/blob/main/mobilesamv2/predictor.py
#   Predict masks for the given input prompts, using the currently set image
def setup_models(args, device):
    """Initialize and setup all required models."""
    clip_model = MaskCLIPFeaturizer().cuda().eval()
    
    mobilesamv2, ObjAwareModel, predictor = torch.hub.load("RogerQi/MobileSAMV2", args.mobilesamv2_encoder_name)
    mobilesamv2.to(device=device)
    mobilesamv2.eval()
    
    return {
        'clip_model': clip_model,
        'mobilesamv2': mobilesamv2,
        'ObjAwareModel': ObjAwareModel,
        'predictor': predictor,
    }

'''
def process_dinov2_features(image_path, dinov2, dino_transform, device, dinov2_feat_path, dino_resolution):
    """Process DINOv2 features for a single image."""
    image = Image.open(image_path)
    image = resize_image(image, dino_resolution)
    image = dino_transform(image)[:3].unsqueeze(0)
    image, target_H, target_W = interpolate_to_patch_size(image, dinov2.patch_size)
    image = image.cuda()
    
    with torch.no_grad():
        features = dinov2.forward_features(image)["x_norm_patchtokens"][0]
    
    features = features.cpu().numpy()
    features_hwc = features.reshape((target_H // dinov2.patch_size, target_W // dinov2.patch_size, -1))
    features_chw = features_hwc.transpose((2, 0, 1))
    
    np.save(dinov2_feat_path, features_chw)
'''

# Object Aware Model is a YOLO model that is used to detect objects in an image
def process_sam_masks(image, ObjAwareModel, predictor, mobilesamv2, device, yolo_conf, yolo_iou, sam_size):
    """Process SAM masks for object detection."""
    obj_results = ObjAwareModel(image, device=device, imgsz=sam_size, conf=yolo_conf, iou=yolo_iou, verbose=False)
    
    predictor.set_image(image)
    input_boxes1 = obj_results[0].boxes.xyxy
    input_boxes = input_boxes1.cpu().numpy()
    input_boxes = predictor.transform.apply_boxes(input_boxes, predictor.original_size)
    input_boxes = torch.from_numpy(input_boxes).cuda()
    
    sam_mask = []
    image_embedding = predictor.features
    image_embedding = torch.repeat_interleave(image_embedding, 320, dim=0)
    prompt_embedding = mobilesamv2.prompt_encoder.get_dense_pe()
    prompt_embedding = torch.repeat_interleave(prompt_embedding, 320, dim=0)
    
    for (boxes,) in batch_iterator(320, input_boxes):
        with torch.no_grad():
            image_embedding = image_embedding[0:boxes.shape[0],:,:,:]
            prompt_embedding = prompt_embedding[0:boxes.shape[0],:,:,:]
            sparse_embeddings, dense_embeddings = mobilesamv2.prompt_encoder(
                points=None,
                boxes=boxes,
                masks=None,)
            low_res_masks, _ = mobilesamv2.mask_decoder(
                image_embeddings=image_embedding,
                image_pe=prompt_embedding,
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
                simple_type=True,
            )
            low_res_masks = predictor.model.postprocess_masks(low_res_masks, predictor.input_size, predictor.original_size)
            sam_mask_pre = (low_res_masks > mobilesamv2.mask_threshold)*1.0
            sam_mask.append(sam_mask_pre.squeeze(1))
    
    return torch.cat(sam_mask), input_boxes1

# get the object-level clip feature
# wrt each pixel, take the average of the clip features of all the objects that overlap with that pixel
def process_object_level_features(image, sam_masks, clip_model, raw_transform, device, 
                                  obj_feat_path, object_H, object_W, final_H, final_W):
    """Process object-level CLIP features."""
    raw_input_image = raw_transform(Image.fromarray(image))
    whole_image_feature = clip_model(raw_input_image[None].cuda())[0]
    clip_feat_shape = whole_image_feature.shape[0]
    
    # Interpolate CLIP features to image size
    resized_clip_feat_map_bchw = torch.nn.functional.interpolate(
        whole_image_feature.unsqueeze(0).float(),
        size=(object_H, object_W),
        mode='bilinear',
        align_corners=False
    )
    
    mask_tensor_bchw = sam_masks.unsqueeze(1)
    resized_mask_tensor_bchw = torch.nn.functional.interpolate(
        mask_tensor_bchw.float(),
        size=(object_H, object_W),
        mode='nearest'
    ).bool()
    
    aggregated_feat_map = torch.zeros((clip_feat_shape, object_H, object_W), dtype=float, device=device)
    aggregated_feat_cnt_map = torch.zeros((object_H, object_W), dtype=int, device=device)
    
    # for each object mask, aggregate the clip features and take the mean of all the pixels in the mask
    # this average of the pixels in the mask is the object-level feature
    # then we interpolate the object-level feature to the final resolution
    # and save the object-level feature
    for mask_idx in range(resized_mask_tensor_bchw.shape[0]):
        aggregared_clip_feat = resized_clip_feat_map_bchw[0, :, resized_mask_tensor_bchw[mask_idx, 0]]
        aggregared_clip_feat = aggregared_clip_feat.mean(dim=1)
        
        aggregated_feat_map[:, resized_mask_tensor_bchw[mask_idx, 0]] += aggregared_clip_feat[:, None]
        aggregated_feat_cnt_map[resized_mask_tensor_bchw[mask_idx, 0]] += 1
    
    # wrt each pixle, take the average of the clip features of all the objects that overlap with that pixel
    # in for loop we get a sum of all the clip features of all the objects that overlap with that pixel
    aggregated_feat_map = aggregated_feat_map / (aggregated_feat_cnt_map[None, :, :] + 1e-6)
    aggregated_feat_map = F.interpolate(
        aggregated_feat_map[None], 
        (final_H, final_W), 
        mode='bilinear', 
        align_corners=False
    )[0]
    
    np.save(obj_feat_path, aggregated_feat_map.cpu().detach().numpy())
    return aggregated_feat_map

def process_object_level_features_mask(image, sam_masks, clip_model, raw_transform, device, 
                                     obj_feat_path, object_H, object_W, final_H, final_W):
    """Process object-level CLIP features where each mask region has its own unique CLIP embedding."""
    raw_input_image = raw_transform(Image.fromarray(image))
    whole_image_feature = clip_model(raw_input_image[None].cuda())[0]
    clip_feat_shape = whole_image_feature.shape[0]
    
    # Interpolate CLIP features to image size
    resized_clip_feat_map_bchw = torch.nn.functional.interpolate(
        whole_image_feature.unsqueeze(0).float(),
        size=(object_H, object_W),
        mode='bilinear',
        align_corners=False
    )
    
    mask_tensor_bchw = sam_masks.unsqueeze(1)
    resized_mask_tensor_bchw = torch.nn.functional.interpolate(
        mask_tensor_bchw.float(),
        size=(object_H, object_W),
        mode='nearest'
    ).bool()
    
    # Create a new feature map where each mask region will have its own unique CLIP embedding
    mask_feat_map = torch.zeros((clip_feat_shape, object_H, object_W), dtype=torch.float32, device=device)
    
    # For each mask, calculate its average CLIP embedding and assign it to the entire mask region
    for mask_idx in range(resized_mask_tensor_bchw.shape[0]):
        # Get the mask region
        mask_region = resized_mask_tensor_bchw[mask_idx, 0]
        
        # Calculate average CLIP embedding for this mask region
        mask_clip_feat = resized_clip_feat_map_bchw[0, :, mask_region]
        mask_avg_feat = mask_clip_feat.mean(dim=1).to(torch.float32)  # Ensure float32 dtype
        
        # Assign this average embedding to the entire mask region
        mask_feat_map[:, mask_region] = mask_avg_feat[:, None]
    
    # Interpolate to final resolution
    mask_feat_map = F.interpolate(
        mask_feat_map[None], 
        (final_H, final_W), 
        mode='bilinear', 
        align_corners=False
    )[0]
    
    np.save(obj_feat_path, mask_feat_map.cpu().detach().numpy())
    return mask_feat_map

def process_part_level_features(image, bbox_xyxy_list, clip_model, part_transform, device, part_feat_path, 
                              small_H, small_W, final_H, final_W, part_batch_size, clip_feat_shape):
    """Process part-level CLIP features."""
    cropped_image_list = []
    for bbox_xyxy in bbox_xyxy_list:
        crop_img = image[bbox_xyxy[1]:bbox_xyxy[3], bbox_xyxy[0]:bbox_xyxy[2]]
        cropped_image_list.append(crop_img)
    
    image_tensor_list = []
    for cropped_image in cropped_image_list:
        if not isinstance(cropped_image, Image.Image):
            cropped_image = Image.fromarray(cropped_image)
        image_tensor = part_transform(cropped_image).unsqueeze(0).to(device)
        image_tensor_list.append(image_tensor)
    
    aggregared_features = []
    for batch_idx in range(0, len(image_tensor_list), part_batch_size):
        with torch.no_grad():
            batch = image_tensor_list[batch_idx:batch_idx+part_batch_size]
            batch = torch.cat(batch, dim=0)
            features = clip_model(batch)
            aggregared_features.append(features)
    
    aggregared_features = torch.cat(aggregared_features, dim=0)
    
    aggregated_feat_map = torch.zeros((clip_feat_shape, small_H, small_W), dtype=float, device=device)
    aggregated_feat_cnt_map = torch.zeros((small_H, small_W), dtype=int, device=device)
    
    for obj_idx in range(len(image_tensor_list)):
        resized_bbox = (bbox_xyxy_list[obj_idx] * (small_W / image.shape[1])).astype(int)
        feat_h = int(resized_bbox[3] - resized_bbox[1])
        feat_w = int(resized_bbox[2] - resized_bbox[0])
        
        resized_feature = F.interpolate(
            aggregared_features[obj_idx].unsqueeze(0), 
            (feat_h, feat_w), 
            mode='bilinear', 
            align_corners=False
        )[0]
        
        aggregated_feat_map[:, resized_bbox[1]:resized_bbox[3], resized_bbox[0]:resized_bbox[2]] += resized_feature
        aggregated_feat_cnt_map[resized_bbox[1]:resized_bbox[3], resized_bbox[0]:resized_bbox[2]] += 1
    
    aggregated_feat_map = aggregated_feat_map / (aggregated_feat_cnt_map[None,:,:] + 1e-6)
    aggregated_feat_map = F.interpolate(
        aggregated_feat_map[None], 
        (final_H, final_W), 
        mode='bilinear', 
        align_corners=False
    )[0]
    
    np.save(part_feat_path, aggregated_feat_map.cpu().numpy())
    return aggregated_feat_map

def save_visualizations(image, sam_mask, input_boxes1, obj_feat_path, aggregated_feat_map, mask_feat_map):
    """Save various visualizations for debugging and analysis."""
    # Save SAM mask visualization
    annotation = sam_mask
    areas = torch.sum(annotation, dim=(1, 2))
    sorted_indices = torch.argsort(areas, descending=True)
    show_img = annotation[sorted_indices]
    ann_img = show_anns(show_img)
    save_img_path = obj_feat_path.replace('.npy', '_mask.png')
    Image.fromarray((ann_img * 255).astype(np.uint8)).save(save_img_path)
    
    # Save bbox visualization
    viz_img = image.copy()
    bboxes_for_save = []
    for bbox_idx in range(input_boxes1.shape[0]):
        bbox = input_boxes1[bbox_idx]
        bbox_xyxy = bbox.cpu().numpy().astype(int)
        bboxes_for_save.append(bbox_xyxy)
        cv2.rectangle(viz_img, (bbox_xyxy[0], bbox_xyxy[1]), (bbox_xyxy[2], bbox_xyxy[3]), (0, 255, 0), 2)
        cv2.putText(viz_img, f'{bbox_idx}', (bbox_xyxy[0], bbox_xyxy[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    save_img_path = obj_feat_path.replace('.npy', '_bbox.png')
    Image.fromarray(viz_img).save(save_img_path)
    
    # Save bounding boxes as .npy
    bbox_save_path = obj_feat_path.replace('.npy', '_bboxes.npy')
    np.save(bbox_save_path, np.stack(bboxes_for_save))
    
    # Save PCA visualization
    original_size = (image.shape[1], image.shape[0])
    visualize_aggregated_feat_map(aggregated_feat_map, obj_feat_path.replace('.npy', '_clip_pca.png'), original_size)
    visualize_aggregated_feat_map(mask_feat_map, obj_feat_path.replace('.npy', '_mask_clip_pca.png'), original_size)
def setup_directories(args):
    """Setup all necessary directories for output."""
    os.makedirs(args.output_path, exist_ok=True)
    args.obj_clip_feat_dir = os.path.join(args.output_path, 'sam_clip_features')
    os.makedirs(args.obj_clip_feat_dir, exist_ok=True)
    args.part_clip_feat_dir = os.path.join(args.output_path, 'part_level_features')
    os.makedirs(args.part_clip_feat_dir, exist_ok=True)
    return args

def get_image_directory(base_dir):
    """Get the image directory path, handling different possible locations."""
    image_dir = os.path.join(base_dir, 'images')
    if not os.path.exists(image_dir):
        image_dir = os.path.join(base_dir, 'color')
    assert os.path.isdir(image_dir), f"Image directory {image_dir} does not exist."
    return image_dir

def get_image_paths(image_dir):
    """Get sorted list of valid image paths from directory."""
    image_paths = [os.path.join(image_dir, fn) for fn in os.listdir(image_dir)]
    image_paths = [fn for fn in image_paths if is_valid_image(fn)]
    image_paths.sort()
    assert len(image_paths) > 0, f"No valid images found in {image_dir}."
    print(f"Found {len(image_paths)} images.")
    return image_paths

def setup_output_paths(image_paths, args):
    """Setup output paths for all features."""
    obj_feat_path_list = []
    part_feat_path_list = []
    
    for image_path in image_paths:
        feat_fn = os.path.splitext(os.path.basename(image_path))[0] + '.npy'
        obj_feat_path = os.path.join(args.obj_clip_feat_dir, feat_fn)
        part_feat_path = os.path.join(args.part_clip_feat_dir, feat_fn)
        obj_feat_path_list.append(obj_feat_path)
        part_feat_path_list.append(part_feat_path)
    
    return obj_feat_path_list, part_feat_path_list

def preprocess_image(image_path, sam_size):
    """Load and preprocess an image for SAM processing."""
    image = cv2.imread(image_path)
    if max(image.shape[:2]) > sam_size:
        if image.shape[0] > image.shape[1]:
            image = cv2.resize(image, (int(sam_size * image.shape[1] / image.shape[0]), sam_size))
        else:
            image = cv2.resize(image, (sam_size, int(sam_size * image.shape[0] / image.shape[1])))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def calculate_dimensions(image_shape, args):
    """Calculate all necessary dimensions for feature processing."""
    raw_img_H, raw_img_W = image_shape[:2]
    
    small_W = args.part_feat_res
    small_H = raw_img_H * small_W // raw_img_W
    
    object_W = args.obj_feat_res
    object_H = raw_img_H * object_W // raw_img_W
    
    final_W = args.final_feat_res
    final_H = raw_img_H * final_W // raw_img_W
    
    return {
        'small_H': small_H,
        'small_W': small_W,
        'object_H': object_H,
        'object_W': object_W,
        'final_H': final_H,
        'final_W': final_W
    }

def process_single_image(image_path, args, models, transforms, device, yolo_conf, yolo_iou, output_paths):
    """Process a single image through all feature extraction pipelines."""
    # Load and preprocess image
    image = preprocess_image(image_path, args.sam_size)
    dims = calculate_dimensions(image.shape, args)
    
    # Process SAM masks
    sam_masks, input_boxes1 = process_sam_masks(
        image, models['ObjAwareModel'], models['predictor'], 
        models['mobilesamv2'], device, yolo_conf, yolo_iou, args.sam_size
    )
    
    # Get CLIP feature shape from a test forward pass
    with torch.no_grad():
        test_input = transforms['raw_transform'](Image.fromarray(image)).unsqueeze(0).cuda()
        clip_feat_shape = models['clip_model'](test_input).shape[1]
    
    # Process object-level features
    aggregated_feat_map = process_object_level_features(
        image, sam_masks, models['clip_model'], transforms['raw_transform'], 
        device, output_paths['obj_feat_path'], dims['object_H'], dims['object_W'],
        dims['final_H'], dims['final_W']
    )

    mask_feat_map = process_object_level_features_mask(image, sam_masks, models['clip_model'], transforms['raw_transform'], device, 
                                     output_paths['obj_feat_path'], dims['object_H'], dims['object_W'], dims['final_H'], dims['final_W'])
    
    # Process part-level features
    bbox_xyxy_list = [bbox.cpu().numpy().astype(int) for bbox in input_boxes1]
    part_aggregated_feat_map = process_part_level_features(
        image, bbox_xyxy_list, models['clip_model'], transforms['part_transform'],
        device, output_paths['part_feat_path'], dims['small_H'], dims['small_W'],
        dims['final_H'], dims['final_W'], args.part_batch_size, clip_feat_shape
    )
    
    # Save visualizations
    save_visualizations(
        image, sam_masks, input_boxes1, 
        output_paths['obj_feat_path'], aggregated_feat_map, mask_feat_map
    )

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    yolo_iou = 0.9
    yolo_conf = 0.4
    
    # Setup all necessary components
    args = setup_directories(args)
    transforms = setup_transforms(args)
    models = setup_models(args, device)
    
    # Get image paths
    image_dir = get_image_directory(args.source_path)
    image_paths = get_image_paths(image_dir)
    obj_feat_path_list, part_feat_path_list = setup_output_paths(image_paths, args)
    
    # Process each image
    for i in trange(len(image_paths)):
        output_paths = {
            'obj_feat_path': obj_feat_path_list[i],
            'part_feat_path': part_feat_path_list[i]
        }
        
        process_single_image(
            image_paths[i], args, models, transforms,
            device, yolo_conf, yolo_iou, output_paths
        )

if __name__ == "__main__":
    parser = ArgumentParser("Compute reference features for feature splatting")
    parser.add_argument("--source_path", "-s", required=True, type=str)
    parser.add_argument("--output_path", "-o", required=True, type=str, help="Directory to save all output files")
    parser.add_argument("--part_batch_size", type=int, default=32, help="Part-level CLIP inference batch size")
    parser.add_argument("--part_resolution", type=int, default=224, help="Part-level CLIP input image resolution")
    parser.add_argument("--sam_size", type=int, default=1024, help="Longest edge for MobileSAMV2 segmentation")
    parser.add_argument("--obj_feat_res", type=int, default=100, help="Intermediate (for MAP) SAM-enhanced Object-level feature resolution")
    parser.add_argument("--part_feat_res", type=int, default=400, help="Intermediate (for MAP) SAM-enhanced Part-level feature resolution")
    parser.add_argument("--final_feat_res", type=int, default=64, help="Final hierarchical CLIP feature resolution")
    parser.add_argument("--mobilesamv2_encoder_name", type=str, default="mobilesamv2_efficientvit_l2", help="MobileSAMV2 encoder name")
    args = parser.parse_args()

    with torch.no_grad():
        main(args)
