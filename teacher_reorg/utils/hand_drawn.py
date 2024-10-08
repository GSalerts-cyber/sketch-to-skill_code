import torch
import numpy as np
from PIL import Image
import h5py
import os

from data_func.dataloader_sketch import Thickening, normalize, standardize


def load_hand_draw_data(img_size):
    root_path = "~/Documents/Teacher_student_RLsketch/teacher_model/sketch3d_cINN_vae/rgb_images"
    sketches1_names = [
        f"{root_path}/demo_0_corner_sketch.tiff", 
        f"{root_path}/demo_6_corner_sketch.tiff", 
    ]
    sketches2_names = [    
        f"{root_path}/demo_0_corner2_sketch.tiff",
        f"{root_path}/demo_6_corner2_sketch.tiff",
    ]
    sketches_names = sketches1_names + sketches2_names
    # load the sketches
    sketches = []
    thickening = Thickening(thickness=4)
    for sketch_path in sketches_names:
        # load the sketch
        sketch = normalize(np.array(Image.open(sketch_path).convert('RGB')))
        # Apply thickening to the sketch
        sketch = thickening(torch.from_numpy(sketch).permute(2, 0, 1)).permute(1, 2, 0).numpy()
        sketches.append(sketch)
    sketches = np.array(sketches) # (img_num, orig_img_size, orig_img_size, 3)

    # Resize the sketches to match the desired image size
    sketches = torch.nn.functional.interpolate(
        torch.from_numpy(sketches).permute(0, 3, 1, 2),
        size=(img_size, img_size),
        mode='bilinear',
        align_corners=False
    )   # (img_num, 3, img_size, img_size)

    # convert to tensor and normalize to [-0.5, 0.5]
    # sketches = torch.tensor(sketches).permute(0, 3, 1, 2).float() - 0.5
    
    # standardize
    data_mean = 0.0037
    data_std = 0.0472
    sketches = (sketches - data_mean) / data_std

    sketches1 = sketches[:len(sketches1_names)]
    sketches2 = sketches[len(sketches1_names):]

    return sketches1, sketches2


def load_hand_draw_data_new(f_name, use_traj_rescale=False, load_traj=False):
    root_path = f"~/hand_drawn_data/sketches/{f_name}"
    sketch1_names = ["demo_0_corner_image_sketch_64",
                     "demo_1_corner_image_sketch_64",
                     "demo_10_corner_image_sketch_64"]
    sketch2_names = ["demo_0_corner2_image_sketch_64",
                     "demo_1_corner2_image_sketch_64",
                     "demo_10_corner2_image_sketch_64"]
    sketches1 = []
    sketches2 = []
    for sketch1_name, sketch2_name in zip(sketch1_names, sketch2_names):
        sketch1 = Image.open(f"{root_path}/{sketch1_name}.png").convert('RGB')
        sketch1 = np.array(sketch1).transpose(2, 0, 1)
        sketches1.append(sketch1)
        sketch2 = Image.open(f"{root_path}/{sketch2_name}.png").convert('RGB')
        sketch2 = np.array(sketch2).transpose(2, 0, 1)
        sketches2.append(sketch2)
    sketches1 = torch.tensor(np.array(sketches1))  
    sketches2 = torch.tensor(np.array(sketches2))

    sketches1 = standardize(normalize(sketches1))
    sketches2 = standardize(normalize(sketches2))

    if use_traj_rescale:
        starts = []
        ends = []
        trajs = []
        root_dir = "~/demo_dataset_bc_96_new"
        file_path = os.path.join(root_dir, f"{f_name}_frame_stack_1_96x96_end_on_success", "dataset.hdf5")
        with h5py.File(file_path, 'r') as f:
            demo_group_key = ["demo_0", "demo_1", "demo_10"]
            for i, demo_name in enumerate(demo_group_key):
                demo_group = f['data'][demo_name]
                obs_group = demo_group['obs']
                state = obs_group['state']
                starts.append(state[0, :3])
                ends.append(state[0, -3:])
                if load_traj:
                    traj = obs_group['prop'][:, :3]
                    trajs.append(traj)
                if i == 2:
                    break
        starts = torch.tensor(np.array(starts))
        ends = torch.tensor(np.array(ends))

        if load_traj:
            return sketches1, sketches2, starts, ends, trajs
        else:
            return sketches1, sketches2, starts, ends
        
    return sketches1, sketches2



if __name__ == "__main__":
    sketches1, sketches2 = load_hand_draw_data(64)
    print(sketches1.shape)
    print(sketches2.shape)
    print(sketches1.dtype)
    print(sketches2.dtype)
