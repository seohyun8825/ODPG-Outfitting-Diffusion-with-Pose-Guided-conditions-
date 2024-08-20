
        axs[4].imshow(sample['img_garment'].permute(1, 2, 0).numpy() * 0.5 + 0.5)
        axs[4].set_title("Garment Image")

        if 'img_cond' in sample:
            axs[5].imshow(sample['img_cond'].permute(1, 2, 0).numpy() * 0.5 + 0.5)
            axs[5].set_title("Conditioned Image")

        if 'masked_img_src' in sample:
            axs[6].imshow(sample['masked_img_src'].permute(1, 2, 0).numpy() * 0.5 + 0.5)
            axs[6].set_title("Masked Src Image")

        if 'masked_img_tgt' in sample:
            axs[7].imshow(sample['masked_img_tgt'].permute(1, 2, 0).numpy() * 0.5 + 0.5)
            axs[7].set_title("Masked Tgt Image")

        for ax in axs:
            ax.axis('off')

        print("HELLO")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{prefix}_{idx}.png"))
        plt.close(fig)

# Example usage
train_dataset = PisTrainDeepFashion(root_dir="/home/user/Desktop/CFLD/CFLD/fashion", gt_img_size=(256, 176), pose_img_size=(256, 176), cond_img_size=(128, 88), min_scale=0.8, log_aspect_ratio=(-0.2, 0.2), pred_ratio=[0.1], pred_ratio_var=[0.05], psz=16)

test_dataset = PisTestDeepFashion(root_dir="/home/user/Desktop/CFLD/CFLD/fashion", gt_img_size=(256, 176), pose_img_size=(256, 176), cond_img_size=(128, 88), test_img_size=(256, 176))

save_dataset_samples(train_dataset, num_samples=2, save_dir='samples', prefix='train')
save_dataset_samples(test_dataset, num_samples=2, save_dir='samples', prefix='test')