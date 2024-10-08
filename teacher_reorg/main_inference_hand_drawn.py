import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import h5py
from data_func.dataloader_helper import normalize, standardize, rescale, Thickening
from model.vae_mlp import VAE_MLP
from utils.hand_drawn import load_hand_draw_data_new


def load_data(f_name):
    img_size = 64
    num_control_points = 20
    root_path = "~/demo_datasets"
    # root_path = "~/sketch_datasets_new"
    sketch1 = torch.load(f'{root_path}/{f_name}/{f_name}_sketches1_{img_size}_cropped.pt')
    sketch2 = torch.load(f'{root_path}/{f_name}/{f_name}_sketches2_{img_size}_cropped.pt')
    traj_gt = torch.load(f'{root_path}/{f_name}/{f_name}_trajectories_raw.pt')
    traj_dense = torch.load(f'{root_path}/{f_name}/{f_name}_fitted_trajectories_50.pt')
    params = torch.load(f'{root_path}/{f_name}/{f_name}_params_{num_control_points}.pt')
    fitted_traj = torch.load(f'{root_path}/{f_name}/{f_name}_fitted_trajectories_{num_control_points}.pt')
    return sketch1, sketch2, traj_gt, traj_dense, params, fitted_traj


def load_model(model_path):
    model_path = f"{model_path}/models/vae_mlp_model_final.pth"
    if "rescale" in model_path:
        model = VAE_MLP(img_size=64, in_channels=3, latent_dim=32, num_control_points=20, degree=3, use_traj_rescale=True).cuda()
    else:
        model = VAE_MLP(img_size=64, in_channels=3, latent_dim=32, num_control_points=20, degree=3).cuda()
    model.load_state_dict(torch.load(model_path))
    return model


def inference(model, sketch1, sketch2, starts, ends, num_gen=10):
    sketch1, sketch2 = sketch1.cuda(), sketch2.cuda()
    print(f"max of sketch1: {torch.max(sketch1)}, min of sketch1: {torch.min(sketch1)}")
    print(f"max of sketch2: {torch.max(sketch2)}, min of sketch2: {torch.min(sketch2)}")
    if ends is not None:
        # Note: apply offset to the z axis of the end points
        ends[:, 2] = ends[:, 2] - 0.05
    with torch.no_grad():
        generated_trajs = []
        for i in range(num_gen):
            recons1, sketch1, mu1, log_var1, recons2, sketch2, mu2, log_var2, params = model(sketch1, sketch2)
            generated_traj = model.generate_trajectory(params)
            if model.use_traj_rescale and starts is not None and ends is not None:
                generated_traj = model.rescale_traj(generated_traj, starts.cuda(), ends.cuda())
            generated_trajs.append(generated_traj.cpu())
    sketch1 = sketch1.cpu()
    sketch2 = sketch2.cpu()
    recons1 = recons1.cpu()
    recons2 = recons2.cpu()
    return sketch1, sketch2, recons1, recons2, generated_trajs



def visualize_trajectory(sketches1, sketches2, recons1, recons2, trajs_gt, generated_trajs, img_name, output_dir):
    rows = 5
    num_samples = len(sketches1)
    fig = plt.figure(figsize=(4 * num_samples, 4 * rows))

    # Create a colormap
    cmap = plt.get_cmap('viridis')

    for i in range(num_samples):
        ax1 = fig.add_subplot(rows, num_samples, i + 1)
        ax1.imshow(rescale(sketches1[i].squeeze()))
        ax1.set_title(f"Sample {i + 1}: Sketch 1")
        ax1.set_aspect('equal', 'box')

        ax2 = fig.add_subplot(rows, num_samples, num_samples + i + 1)
        ax2.imshow(rescale(recons1[i].squeeze()))
        ax2.set_title(f"Recon1 MSE: {np.mean((sketches1[i] - recons1[i])**2):.6f}")
        ax2.set_aspect('equal', 'box')

        ax3 = fig.add_subplot(rows, num_samples, 2 * num_samples + i + 1)
        ax3.imshow(rescale(sketches2[i].squeeze()))
        ax3.set_title(f"Sample {i + 1}: Sketch 2")
        ax3.set_aspect('equal', 'box')

        ax4 = fig.add_subplot(rows, num_samples, 3 * num_samples + i + 1)
        ax4.imshow(rescale(recons2[i].squeeze()))
        ax4.set_title(f"Recon2 MSE: {np.mean((sketches2[i] - recons2[i])**2):.6f}")
        ax4.set_aspect('equal', 'box')

        ax5 = fig.add_subplot(rows, num_samples, 4 * num_samples + i + 1, projection='3d')
        for generated_traj in generated_trajs:
            ax5.plot(generated_traj[i][:, 0], generated_traj[i][:, 1], generated_traj[i][:, 2], alpha=0.5, linewidth=2)
        ax5.scatter(trajs_gt[i][:, 0], trajs_gt[i][:, 1], trajs_gt[i][:, 2], c=np.arange(len(trajs_gt[i])), cmap=cmap, s=5)
        ax5.set_xlabel('X')
        ax5.set_ylabel('Y')
        ax5.set_zlabel('Z')
        ax5.set_title(f"Generated Traj")
        ax5.set_box_aspect((1, 1, 1))
        ax5.set_aspect('equal')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/{img_name}.png')
    plt.close()


def save_trajectory(generated_trajs, file_name, output_dir):
    # Check if generated_trajs is of length 3
    if generated_trajs[0].shape[0] != 3:
        print(f"Warning: Expected 3 generated trajectories, but got {len(generated_trajs)}")
        return

    traj_dict = {
        "demo_0": np.array([gen_traj[0] for gen_traj in generated_trajs]),
        "demo_1": np.array([gen_traj[1] for gen_traj in generated_trajs]),
        "demo_10": np.array([gen_traj[2] for gen_traj in generated_trajs]),
    }

    with h5py.File(f'{output_dir}/{file_name}.hdf5', 'w') as f:
        for key, value in traj_dict.items():
            f.create_dataset(key, data=value)
    print(f"Trajectory saved to {output_dir}/{file_name}.hdf5")


def inference_and_visualize(model, data_file_name, save_path):
    use_traj_rescale = True
    sketches1, sketches2, starts, ends, trajs_gt = load_hand_draw_data_new(data_file_name, use_traj_rescale, load_traj=True)
    sketch1, sketch2, recons1, recons2, generated_trajs = inference(model, sketches1, sketches2, starts, ends)
    print("Number of samples: ", sketch1.shape[0])

    sketch1 = sketch1.numpy().transpose(0, 2, 3, 1)
    sketch2 = sketch2.numpy().transpose(0, 2, 3, 1)
    recons1 = recons1.numpy().transpose(0, 2, 3, 1)
    recons2 = recons2.numpy().transpose(0, 2, 3, 1)
    generated_trajs = [traj.numpy() for traj in generated_trajs]

    visualize_trajectory(sketch1, sketch2, recons1, recons2, trajs_gt, generated_trajs, f"generated_trajectory_{data_file_name}_inference_hand_drawn", save_path)
    save_trajectory(generated_trajs, f"generated_trajectory_{data_file_name}_inference_hand_drawn", save_path)
    

if __name__ == "__main__":
    # model_pathes = ['~/VAE_MLP_Both_NewData2/vae_mlp_ep200_bs128_dim32_cosine_lr0.001_kld0.0001_aug_2024-09-26_13-58-16',
    #                 '~/VAE_MLP_Both_NewData2/vae_mlp_ep200_bs128_dim32_cosine_lr0.001_kld0.1_aug_2024-09-26_13-58-25',
    #                 '~/VAE_MLP_Both_NewData2/vae_mlp_ep200_bs128_dim32_cosine_lr0.001_kld0.001_aug_2024-09-26_13-58-25',
    #                 '~/VAE_MLP_Both_NewData2/vae_mlp_ep200_bs128_dim32_cosine_lr0.001_kld0.0001_aug_Anneal_2024-09-26_13-58-22',
    #                 '~/VAE_MLP_Both_NewData2/vae_mlp_ep200_bs128_dim32_cosine_lr0.001_kld0.001_aug_Anneal_2024-09-26_13-58-25',
    #                 '~/VAE_MLP_Both_NewData2/vae_mlp_ep200_bs128_dim32_cosine_lr0.001_kld0.1_aug_Anneal_2024-09-26_13-58-31',
    #                 '~/VAE_MLP_Both_NewData2/vae_mlp_ep200_bs128_dim32_cosine_lr0.001_kld0.0005_aug_2024-09-26_13-58-22',
    #                 '~/VAE_MLP_Both_NewData2/vae_mlp_ep200_bs128_dim32_cosine_lr0.001_kld0.0005_aug_Anneal_2024-09-26_13-58-25',
    #                 '~/VAE_MLP_Both_NewData2/vae_mlp_ep200_bs128_dim32_cosine_lr0.001_kld0.00025_aug_2024-09-26_13-58-22',
    #                 '~/VAE_MLP_Both_NewData2/vae_mlp_ep200_bs128_dim32_cosine_lr0.001_kld0.00025_aug_Anneal_2024-09-26_13-58-22',
    #                 '~/VAE_MLP_Both_NewData2/vae_mlp_ep200_bs256_dim32_cosine_lr0.001_kld0.0001_aug_2024-09-26_13-58-09',
    #                 '~/VAE_MLP_Both_NewData2/vae_mlp_ep200_bs256_dim32_cosine_lr0.001_kld0.001_aug_2024-09-26_13-58-13',
    #                 '~/VAE_MLP_Both_NewData2/vae_mlp_ep200_bs256_dim32_cosine_lr0.001_kld0.1_aug_2024-09-26_13-58-16',
    #                 '~/VAE_MLP_Both_NewData2/vae_mlp_ep200_bs256_dim32_cosine_lr0.001_kld0.0001_aug_Anneal_2024-09-26_13-58-09',
    #                 '~/VAE_MLP_Both_NewData2/vae_mlp_ep200_bs256_dim32_cosine_lr0.001_kld0.1_aug_Anneal_2024-09-26_13-58-16',
    #                 '~/VAE_MLP_Both_NewData2/vae_mlp_ep200_bs256_dim32_cosine_lr0.001_kld0.001_aug_Anneal_2024-09-26_13-58-16',
    #                 '~/VAE_MLP_Both_NewData2/vae_mlp_ep200_bs256_dim32_cosine_lr0.001_kld0.0005_aug_2024-09-26_13-58-13',
    #                 '~/VAE_MLP_Both_NewData2/vae_mlp_ep200_bs256_dim32_cosine_lr0.001_kld0.0005_aug_Anneal_2024-09-26_13-58-13',
    #                 '~/VAE_MLP_Both_NewData2/vae_mlp_ep200_bs256_dim32_cosine_lr0.001_kld0.00025_aug_2024-09-26_13-58-09',
    #                 '~/VAE_MLP_Both_NewData2/vae_mlp_ep200_bs256_dim32_cosine_lr0.001_kld0.00025_aug_Anneal_2024-09-26_13-58-13',]
    # model_pathes = ['~/VAE_MLP_Both_NewData2/vae_mlp_ep200_bs128_dim32_cosine_lr0.001_kld0.001_aug_2024-09-26_13-58-25 [bad at ButtonPressTopDownWall, others are reasonable]',
    #                 '~/VAE_MLP_Both_NewData2/vae_mlp_ep200_bs256_dim32_cosine_lr0.001_kld0.001_aug_2024-09-26_13-58-13 [bad at ButtonPressTopDownWall, others are reasonable]',
    #                 '~/VAE_MLP_Both_NewData2/vae_mlp_ep200_bs128_dim32_cosine_lr0.001_kld0.0005_aug_2024-09-26_13-58-22 [somewhat better]',]
    
    model_pathes = ['~/VAE_MLP_Both_NewData2_0927_rescale/vae_mlp_ep200_bs128_dim32_cosine_lr0.00100_kld0.00010_aug_2024-09-27_06-44-04',]
    
    data_file_names = ['ButtonPress',]
                    # 'ButtonPressTopdownWall',
                    # 'DrawerOpen',
                    # 'Reach',
                    # 'ReachWall',]
    
    for model_path in model_pathes:
        print(f"Model path: {model_path}")
        model = load_model(model_path)
        for data_file_name in data_file_names:
            print(f"Data file name: {data_file_name}")
            inference_and_visualize(model, data_file_name, model_path)
            # breakpoint()