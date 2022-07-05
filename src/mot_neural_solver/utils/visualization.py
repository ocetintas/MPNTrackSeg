import numpy as np
import matplotlib.pyplot as plt

def plot_box_and_mask(img, boxes, masks, file_name):
    fig, ax = plt.subplots(1, dpi=96)

    img = img.mul(255).permute(1, 2, 0).byte().numpy()
    width, height = img.shape[0:2]

    ax.imshow(img)
    fig.set_size_inches(width / 80, height / 80)

    colors = np.random.rand(len(boxes), 3)

    for i, box in enumerate(boxes):
        rect = plt.Rectangle(
            (box[0], box[1]),
            box[2] - box[0],
            box[3] - box[1],
            fill=False,
            color=colors[i],
            linewidth=1.0)
        ax.add_patch(rect)

        mask = masks[i]

        # add mask
        for channel in range(3):
            img[:, :, channel] = np.where(mask == 1, img[:, :, channel] * 0.3 + 0.7 * colors[i][channel] * 255,
                                          img[:, :, channel])

    ax.imshow(img)
    plt.axis('off')
    plt.savefig(file_name + '.png')