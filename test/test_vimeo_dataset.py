import sys
import os
import os.path as osp
import matplotlib.pyplot as plt

sys.path.insert(0, osp.join(os.getcwd(), '../src'))
os.chdir(osp.join(os.getcwd(), '../src'))
from data.vimeo90k_dataset import Vimeo90kDataset


def visualize(mode, num_samples):
    dataset = Vimeo90kDataset(mode)
    figure, ax = plt.subplots(nrows=num_samples, ncols=3, figsize=(40, 40))

    print(len(dataset))

    for idx in range(num_samples):
        img, img_color_transformed, img_spatial_transformed = dataset[idx]
        ax.ravel()[3 * idx].imshow(img)
        ax.ravel()[3 * idx + 1].imshow(img_color_transformed)
        ax.ravel()[3 * idx + 2].imshow(img_spatial_transformed)
        ax.ravel()[idx].set_axis_off()
    plt.tight_layout()
    plt.savefig(f'../test/out/vimeo_dataset/{mode}.png')


if __name__ == '__main__':
    visualize('train', 5)
    visualize('test', 5)
