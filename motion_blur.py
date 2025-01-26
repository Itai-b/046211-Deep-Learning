import os
import cv2
import torch
import torchvision
import numpy as np
import random as rnd
import matplotlib.pyplot as plt


def uniform_kernel(kernel_size=50, thickness=0, angle=45):
    """
    Generate a uniform motion blur kernel of given size and angle.

    Parameters:
    - kernel_size (int): size of the motion blur kernel (controls blur length).
    - angle (int): Angle of motion blur in degrees.
    - thickness (int): Thickness of the PSF kernel (controls blur width).
    - mode (str): Mode of the PSF kernel ('full', 'half_right', 'half_left').

    Returns:
    - Uniform motion blur kernel (numpy.ndarray).
    """
    kernel = np.zeros((kernel_size, kernel_size))
    center = kernel_size // 2
    #for i in range(kernel_size):
    #    kernel[center, i] = 1

    # Draw a thicker line using cv2.line
    start_point = (0, center - thickness // 2)
    end_point = (kernel_size - 1, center - thickness // 2)
    cv2.line(kernel, start_point, end_point, 1, thickness)

    # Normalize kernel
    if np.sum(kernel) > 0:
        kernel /= np.sum(kernel)

    # Rotate kernel to desired angle
    rotation_matrix = cv2.getRotationMatrix2D((center, center), angle, 1)
    kernel = cv2.warpAffine(kernel, rotation_matrix, (kernel_size, kernel_size))

    return kernel


def apply_uniform_motion_blur(image, amplitude=0.5):
    """
    Apply a uniform motion blur to the image.

    Parameters:
    - image (numpy.ndarray) - the image to apply the kernel on.
    - amplitude (float) - the amplitude (severity) of the blur. Range: [0,1]

    Returns:
    - Motion blurred image and kernel (numpy.ndarray, numpy.ndarray).
    """
    # Ensure sigma is within valid bounds
    if amplitude <= 0:
        return image, np.zeros((1, 1))
    if amplitude > 1:
        amplitude = 1

    # Set up kernel size based on the amplitude of the motion blur
    min_size = 2
    max_size = min(image.shape[0], image.shape[1])     # take the minimum between image height and width
    size = int(min_size + amplitude * (max_size - min_size))

    # Generate random values for thickness (limited to small thickness due to the sparse nature of motion blur kernels)
    thickness = rnd.randint(1, max(1, size//20))

    # Generate random angle for the uniform motion blur
    angle = rnd.randint(0, 180)

    # Apply kernel on image
    kernel = uniform_kernel(kernel_size=size, thickness=thickness, angle=angle)
    #blurred_image = cv2.filter2D(image, -1, kernel)

    blurred_image = np.zeros_like(image)
    for c in range(image.shape[2]):  # Iterate over the image's channels
        blurred_image[:, :, c] = cv2.filter2D(image[:, :, c], -1, kernel)

    return blurred_image, kernel


def ellipse_kernel(kernel_size=50, thickness=0, angle=42, mode='full'):
    """
    Generates an ellipse-shaped point spread function (PSF) as a kernel for motion blur.

    Parameters:
    - kernel_size (int): Size of the PSF kernel (controls blur length).
    - thickness (int): Thickness of the PSF kernel (controls blur width).
    - angle (int): Angle of motion blur in degrees.
    - mode (str): Mode of the PSF kernel ('full', 'half_right', 'half_left').

    Returns:
    - Ellipse motion blur kernel (numpy.ndarray).
    """

    # Create an empty Point Spread Function (PSF) kernel (3 channels for RGB, later will be set to (1,1,1)
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    center = (kernel_size // 2, kernel_size // 2)

    # Define the axes of the ellipse (major "radius" and minor "radius")
    # Since the ellipse is filled, these parameters define how long the ellipse will stretch
    # and how high it will reach, where axes=(kernel_size//2, kernel_size//2) is a full circle kernel
    # Simply put, axes=(blur length, PSF thickness)
    axes = (kernel_size // 2, thickness)

    # Define how the kernel should look like ('full' = [1111], 'half_right' = [0011], 'half_left' = [1100])
    start_angle, end_angle = 0, 360 # Default for 'full' mode
    if mode == 'half_right':
        end_angle = 90
    elif mode == 'half_left':
        start_angle = 90
        end_angle = 180

    # Define the PSF kernel using the ellipse function (drawing an ellipse)
    kernel = cv2.ellipse(img=kernel,
                         center=center,            # center of the ellipse (x,y)
                         axes=axes,                # axes of the ellipse
                         angle=angle,              # angle of motion in degrees
                         startAngle=start_angle,   # start angle of the ellipse
                         endAngle=end_angle,       # end angle of the ellipse (0-360 for full ellipse, not an arc)
                         color= 1,         # white color (R, G, B)
                         thickness=-1)             # filling thickness (not the same as the other thickness!)

    # normalize by sum of one channel (since channels are processed independently)
    if np.sum(kernel) > 0:
        kernel /= np.sum(kernel)

    return kernel


def apply_ellipse_motion_blur(image, amplitude=0.5):
    """
    Applies an ellipse-shaped point spread function (PSF) to an image to create a motion blur effect.

    Parameters:
    - image (numpy.ndarray) - the image to apply the kernel on.
    - amplitude (float) - the amplitude (severity) of the blur. Range: [0,1]

    Returns:
    - Motion blurred image and kernel (numpy.ndarray, numpy.ndarray).
    """
    # Ensure sigma is within valid bounds
    if amplitude <= 0:
        return image, np.zeros((1,1))
    if amplitude > 1:
        amplitude = 1

    # Set up kernel size based on the amplitude of the motion blur
    min_size = 2
    max_size = min(image.shape[0], image.shape[1])  # take the minimum between image height and width
    size = int(min_size + amplitude * (max_size - min_size))

    # Generate random values for thickness (limited to small thickness due to the sparse nature of motion blur kernels)
    thickness = rnd.randint(0, max(1, size // 20))

    # Generate random angle for the uniform motion blur
    angle = rnd.randint(0, 180)

    # Randomize mode (full / half kernel)
    mode = rnd.choices(population=['full','half_right','half_left'],weights=[0.5, 0.25,0.25])[0]

    # Apply kernel on image
    kernel = ellipse_kernel(kernel_size=size, thickness=thickness, angle=angle, mode=mode)
    # blurred_image = cv2.filter2D(image, -1, kernel)
    blurred_image = np.zeros_like(image)
    for c in range(image.shape[2]):  # Iterate over the image's channels
        blurred_image[:, :, c] = cv2.filter2D(image[:, :, c], -1, kernel)

    return blurred_image, kernel


def apply_camera_shake_blur(image, kernel_path='./motion_blur_kernels'):
    """
    Applies a randomly selected naturally recorded camera shake blur kernel to an RGB image.

    Parameters:
        image (numpy.ndarray): The input RGB image (H, W, C=3). (height, width, 3 channels for RGB)
        kernel_path (str): Path to the folder containing the recorded kernel images.

    Returns:
       Motion blurred image and kernel (numpy.ndarray, numpy.ndarray).
    """
    # Get a list of all kernel files
    kernel_files = [f for f in os.listdir(kernel_path) if f.endswith('.png')]

    if not kernel_files:
        raise FileNotFoundError(f"No kernel files found in {kernel_path}.")

    # Select a random kernel
    selected_kernel_file = rnd.choice(kernel_files)
    kernel_full_path = os.path.join(kernel_path, selected_kernel_file)

    # Load the kernel as a grayscale image and normalize it
    kernel = cv2.imread(kernel_full_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
    if kernel is None:
        raise ValueError(f"Failed to load kernel image: {kernel_full_path}")

    # Normalize kernel
    if np.sum(kernel) > 0:
        kernel /= np.sum(kernel)

    # Apply the kernel to each channel of the RGB image
    image_filtered = np.zeros_like(image)
    for c in range(image.shape[2]):  # Iterate over the RGB channels
        image_filtered[:, :, c] = cv2.filter2D(image[:, :, c], -1, kernel)

    return image_filtered, kernel


def motion_blur(image, kernel_type='uniform', a=0.2, return_kernel=False):
    """
    Applies motion blur to an image using a generated PSF kernel.

    Parameters:
    - image (numpy.ndarray / torch.Tensor): Input image.
    - sigma (float): Standard deviation controlling the blur intensity. Range: [0, 1].
    - kernel_type (string): Type of blur kernel (uniform synthesized, ellipse synthesized, or real-life recorded).
    - return_kernel (bool): True to also return the kernel used.

    Returns:
    - Motion-blurred image (numpy.ndarray / torch.Tensor - depends on the input)
    - Motion blur kernel used (numpy.ndarray, ONLY WHEN return_kernel=True)
    """
    # Convert to numpy array if the image is in Tensor format:
    if isinstance(image, torch.Tensor):
        image = image.permute(1, 2, 0).numpy()  # Convert from CxHxW to HxWxC
        tensor_format = True
    else:
        tensor_format = False
    
    if kernel_type == 'uniform':
        blurred_image, kernel = apply_uniform_motion_blur(image=image, amplitude=a)
    elif kernel_type == 'ellipse':
        blurred_image, kernel = apply_ellipse_motion_blur(image=image, amplitude=a)
    elif kernel_type == 'natural' or kernel_type == 'camera_shake':
        blurred_image, kernel = apply_camera_shake_blur(image=image)
    else:   # Invalid type, return original image with no kernel
        blurred_image, kernel = (image, np.zeros((1,1)))

    # Post-process the data and return np.array / tensor depends on the input
    if tensor_format:   # working with tensor input, returning a tensor output
        blurred_image = torchvision.transforms.ToTensor()(blurred_image)

    if return_kernel:
        return blurred_image, kernel
    return blurred_image


def show_kernel(kernel):
    """Print the kernel as a heatmap."""
    plt.imshow(kernel, cmap='gray')
    plt.colorbar()
    plt.title("Motion Blur Kernel")
    plt.show()


def test():
    """
    Demonstrates the motion_blur function on 3 random images.
    Displays original and blurred images with their respective kernels.
    """
    # Paths
    images_path = './images'
    image_files = [f for f in os.listdir(images_path) if f.endswith(('.jpg', '.png'))]

    if len(image_files) < 3:
        print("Please ensure at least 3 images are available in the './images' directory.")
        return

    # Select 3 random images
    selected_images = rnd.sample(image_files, 3)

    # Set sigma values for demonstration
    sigma_values = [0, 0.5, 1]

    # Create a matplotlib figure
    fig, axes = plt.subplots(len(selected_images) * 2, len(sigma_values), figsize=(15, 10))

    # set the figure position and size
    manager = plt.get_current_fig_manager()
    manager.window.setGeometry(100, 100, 1200, 800)

    for row_idx, image_file in enumerate(selected_images):
        image_path = os.path.join(images_path, image_file)
        image = cv2.imread(image_path)

        if image is None:
            print(f"Error: Unable to read image {image_path}")
            continue

        # Process the image with different sigma values
        for col_idx, sigma in enumerate(sigma_values):
            # TODO: add plots for both functions
            filtered_image, kernel = motion_blur(image, 'ellipse', a=0.02, return_kernel=True)
            #filtered_image, kernel = apply_motion_blur_kernel(image)

            # Display the image
            ax_image = axes[row_idx * 2, col_idx]
            ax_image.imshow(cv2.cvtColor(filtered_image, cv2.COLOR_BGR2RGB))
            ax_image.set_title(f'{image_file} (σ={sigma})')
            ax_image.axis('off')

            # Normalize and display the kernel (handle case of sigma=0 kernel (which is np.zeros((1,1)))
            if kernel.max() > 0:
                kernel_normalized = (kernel / kernel.max() * 255).astype(np.uint8)
            else:
                kernel_normalized = np.zeros_like(kernel, dtype=np.uint8)
            ax_kernel = axes[row_idx * 2 + 1, col_idx]
            ax_kernel.imshow(kernel_normalized, cmap='gray')
            ax_kernel.set_title(f'Kernel (σ={sigma})')
            ax_kernel.axis('off')

    # Adjust layout
    plt.tight_layout()
    plt.show()


def main():
    """
    Take one image (img-290.jpg) and plot it in 9 variations (3 for each noise)
    """
    # Constants
    image_path= './images/img-290.jpg'
    kernel_types = ['natural', 'uniform', 'ellipse']

    # Set different noise amplitude values for demonstration
    blur_amps = [0.05, 0.1, 0.2]

    # Import the wanted image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to read image {image_path}")
        return -1

    # Define figure and axes and set the figure position and size
    fig, axes = plt.subplots(3,3, figsize=(10,15))
    fig.suptitle("Motion Blur Noise Variations", fontsize=16, weight='bold')
    #manager = plt.get_current_fig_manager()
    #manager.window.setGeometry(100, 100, 1200, 800)

    for row, kernel_type in enumerate(kernel_types):
        for col, amp in enumerate(blur_amps):
            blurred_image, kernel = motion_blur(image, kernel_type, amp, return_kernel=True)

            ax = axes[row, col]
            ax.imshow(cv2.cvtColor(blurred_image, cv2.COLOR_BGR2RGB))
            ax.axis('off')

            if row != 0:    # Natural noise has no defined
                ax.set_title(f'Noise amplitude: {amp}', fontsize=12, weight='bold')

            # Add kernel inset
            inset_ax = ax.inset_axes([-0.085, 0.6, 0.4, 0.4])  # Top-left corner
            inset_ax.imshow(kernel, cmap='gray')
            inset_ax.axis('off')
            #inset_ax.set_title(f"Kernel", fontsize=8)

        # Add row labels
        axes[row, 0].text(-50, image.shape[0] // 2, f"{kernel_type.capitalize()} Kernel",
                          fontsize=14, weight='bold', va='center', rotation=90)

    # Adjust layout
    #plt.tight_layout(rect=[0, 0, 1, 0.95])
    global i
    #plt.tight_layout()
    plt.savefig(os.path.join('./results/', f'res_{i}.png'), dpi=3000) #, bbox_inches='tight')
    plt.show()


i = 2
# This ensures the main function (main block) will not run when motion_blur.py is imported
# If this file is imported then its name is *not* __main__, because it's not the main entry point, so the main() function won't be called
if __name__ == "__main__":
    main()
