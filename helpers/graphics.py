import matplotlib.pyplot as plt


def display_images(images, image_ids, gallery_size=3, figure_size=10):
    """
    Displays a gallery of pictures

    Input:
    images, image_ids - array of images of shape (m, n_H, n_W, n_C) and corresponding ids
    gallery_size - number of images per row to be displayed (total size is gallery_width x gallery_width)
    figure_size - size of images to be displayed
    """

    plt.figure(figsize=(figure_size, figure_size))
    for i in range(gallery_size ** 2):
        subplot = plt.subplot(gallery_size, gallery_size, i + 1)
        plt.imshow(images[i].astype("uint8"))
        plt.title("id=" + str(image_ids[i]))
        plt.axis("off")


def visualize_segmented(before, after, figure_size=10):
    """
    Function that visualizes a pair. First image in pair is before segmenting and another one after the segmenting
    """

    plt.figure(figsize=(figure_size, figure_size))

    subplot = plt.subplot(1, 2, 1)
    plt.imshow(before.astype("uint8"))
    plt.title("Before segmenting")
    plt.axis("off")

    subplot = plt.subplot(1, 2, 2)
    plt.imshow(after.astype("uint8"))
    plt.title("After segmenting")
    plt.axis("off")


def distance_visualize(image1, image2, distance, distance_label=''):
    """
    Function that shows two images and distance between them on the same plot

    Input:
    image1, image2 - two images
    distance - distance between images

    """

    plt.figure(figsize=(5, 2))
    # Drawing first picture
    ax = plt.subplot(1, 3, 1)
    plt.imshow(image1.astype("uint8"))
    plt.axis("off")
    # Drawing second picture
    ax = plt.subplot(1, 3, 2)
    plt.imshow(image2.astype("uint8"))
    plt.axis("off")
    # Drawing a distance between them
    ax = plt.subplot(1, 3, 3)
    plt.text(0.5, 0.5,
             str(distance) if len(distance_label) == 0 else distance_label + '=' + str(distance),
             fontsize='xx-large')
    plt.axis("off")


def draw_plot(X, Y, X_label, Y_label, title, color):
    # Setting a style to a graph
    plt.style.use('default')

    # Plotting the values and adding labels
    plt.plot(X, Y, color=color)
    plt.xlabel(X_label)
    plt.ylabel(Y_label)
    plt.title(title)

    plt.show()
