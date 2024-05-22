import matplotlib.pyplot as plt
import os


def plot_loss(data1, data2, image_folder, image_name):
    # Plotting the first line
    plt.plot(data1, label='train loss')

    # Plotting the second line
    plt.plot(data2, label='test loss')

    # Adding labels and title
    plt.xlabel('X-axis Label')
    plt.ylabel('Y-axis Label')
    plt.title('train-test loss Plot')

    # Adding a legend
    plt.legend()

    # Saving the figure to the specified file path
    plt.savefig(os.path.join(image_folder, image_name))

    # Displaying the plot (optional)
    plt.show()

