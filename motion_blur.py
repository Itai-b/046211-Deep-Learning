import os
import cv2
import torch
import torchvision
import numpy as np
import random as rnd
import matplotlib.pyplot as plt

# REQUIRED DEPENDENCIES: $ sudo apt-get install libmagickwand-dev
# from wand.image import Image

# TODO: sigma affects randomization range and is not a rating for the added noise as in HW2 EX3 (end of notebook)
#       maybe use PSNR (peak signal noise ratio) instead of sigma?
# TODO: fix issue with small sigma size resulting in < 1 kernel size and other params
def apply_motion_blur_psf(image, kernel_size=50, thickness=0, angle=42, mode='full'):
    """
    Applies an ellipse-shaped point spread function (PSF) to an image to create a motion blur effect.

    Parameters:
    - image (numpy.ndarray): Input image.
    - kernel_size (int): Size of the PSF kernel (controls blur length).
    - thickness (int): Thickness of the PSF kernel (controls blur width).
    - angle (int): Angle of motion blur in degrees.
    - mode (str): Mode of the PSF kernel ('full', 'half_right', 'half_left').

    Returns:
    - (numpy.ndarray, numpy.ndarray): Motion-blurred image and the PSF kernel.
    """
    # if kernel is 1x1 no action is needed
    if kernel_size == 1:
        return image, np.zeros((1,1))

    # Create an empty Point Spread Function (PSF) kernel (3 channels for RGB, later will be set to (1,1,1)
    psf = np.zeros((kernel_size, kernel_size, 3), dtype=np.float32)
    center = (kernel_size // 2, kernel_size // 2)

    # Define the axes of the ellipse (major "radius" and minor "radius")
    # Since the ellipse is filled, these parameters define how long the ellipse will stretch
    # and how high it will reach, where axes=(kernel_size//2, kernel_size//2) is a full circle kernel
    # Simply put, axes=(blur length, PSF thickness)
    axes = (kernel_size // 4, thickness)

    # Define how the kernel should look like ('full' = [1111], 'half_right' = [0011], 'half_left' = [1100])
    start_angle, end_angle = 0, 360 # Default for 'full' mode
    if mode == 'half_right':
        end_angle = 90
    elif mode == 'half_left':
        start_angle = 90
        end_angle = 180

    # Define the PSF kernel using the ellipse function (drawing an ellipse)
    psf = cv2.ellipse(img=psf,
                      center=center,            # center of the ellipse (x,y)
                      axes=axes,                # axes of the ellipse
                      angle=angle,              # angle of motion in degrees
                      startAngle=start_angle,   # start angle of the ellipse
                      endAngle=end_angle,       # end angle of the ellipse (0-360 for full ellipse, not an arc)
                      color= (1, 1, 1),         # white color (R, G, B)
                      thickness=-1)             # filled (not the same as the axes thickness!)

    # normalize by sum of one channel (since channels are processed independently)
    psf_sum = psf[:, :, 0].sum()
    if psf_sum > 0:
        psf = psf / psf_sum

    image_filtered = cv2.filter2D(image, -1, psf)
    return image_filtered, psf


def generate_random_params(sigma=0.5, max_kernel_size = 100, min_kernel_size = 1):
    """
    Generate random parameters for the apply_motion_blur_psf function's ellipse kernel
    The randomized values are based on the standard deviation 'sigma' param.

    Parameters:
    - sigma (float): Standard deviation controlling the blur intensity. Range: [0, 1]
                     Higher sigma leads to more intense motion blur effects.
    - max_kernel_size (uint): Maximum size of kernel
    - min_kernel_size (uint): Minimum size of kernel

    Returns:
    - dict: A dictionary of randomly generated parameters for the apply_motion_blur_psf function.
    """
    # Ensure sigma is within valid bounds
    if sigma < 0:
        sigma = 0
    if sigma > 1:
        sigma = 1

    # Ensure kernel size has valid bounds
    if min_kernel_size < 1:
        min_kernel_size = 1

    # Kernel size: Represents the length of the motion blur kernel
    #              Larger (longer) kernel for higher sigma values
    kernel_mean = (max_kernel_size - min_kernel_size) // 2
    kernel_std = 20     # standard deviation for kernel size
    #kernel_size = rnd.randint(min_kernel_size, max(1, int(sigma * max_kernel_size)))
    kernel_size = int(max(1, min(max_kernel_size, round(rnd.gauss(kernel_mean * sigma, kernel_std * sigma)))))

    # Thickness: Represents the width of the motion blur kernel.
    #            Should be small relative to kernel_size to ensure the blur looks like a motion (and not regular blur)
    max_thickness = max(1, kernel_size // 20)
    thickness = rnd.randint(0, max_thickness)

    # Angle: Represents the direction of the motion blur in degrees
    #        Range: [0, 180)
    angle = rnd.randint(0, 180)

    # Mode: Represents the shape of the blur kernel (full kernel / half kernel + which side)
    modes = ['full', 'half_right', 'half_left']
    mode = 'full'
    if not (sigma == 0 or kernel_size < 4):
        mode = rnd.choices(modes)[0]

    psf_params = {
        'kernel_size': kernel_size,
        'thickness': thickness,
        'angle': angle,
        'mode': mode
    }
    return psf_params


def apply_motion_blur_kernel(image, kernel_path='./motion_blur_kernels'):
    """
    Applies a randomly selected motion blur kernel to an RGB image.

    Parameters:
        image (numpy.ndarray): The input RGB image (H, W, C=3). (height, width, 3 channels for RGB)
        kernel_path (str): Path to the folder containing the kernel images.

    Returns:
        numpy.ndarray: The image with the applied motion blur.
        numpy.ndarray: The kernel used for the convolution.
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

    if np.sum(kernel) > 0:
        kernel /= np.sum(kernel)  # Normalize kernel to preserve image intensity

    # Apply the kernel to each channel of the RGB image
    image_filtered = np.zeros_like(image)
    for c in range(3):  # Iterate over the RGB channels
        image_filtered[:, :, c] = cv2.filter2D(image[:, :, c], -1, kernel)

    return image_filtered, kernel


def motion_blur(image, synth=True, sigma=0.5, return_kernel=False):
    """
    Applies motion blur to an image using a generated PSF kernel.

    Parameters:
    - image (numpy.ndarray / torch.tensor): Input image.
    - sigma (float): Standard deviation controlling the blur intensity. Range: [0, 1].
    - synth (bool): Controls the type of motion blur to apply (synthesized kernel or real-life recorded kernel).
    - return_kernel (bool): Controls if the function will also return the kernel it used or not.

    Returns:
    - Motion-blurred image (numpy.ndarray / torch.tensor - depends on the input)
    - Motion blur kernel used (numpy.ndarray, ONLY WHEN return_kernel=True)
    """
    # If image is in tensor format:
    if isinstance(image, torch.Tensor):
        image = image.permute(1, 2, 0).numpy()  # Convert from CxHxW to HxWxC
        tensor_format = True
    else:
        tensor_format = False
    
    if synth:   # synthesized motion blur kernels
        # Generate random parameters for the motion blur function's kernel (PSF)
        #psf_parameters = generate_random_params(sigma)
        image_height, image_width, _ = image.shape
        psf_parameters = generate_random_params(sigma=sigma, max_kernel_size=min(image_height, image_width))
        filtered_image, kernel = apply_motion_blur_psf(
            image=image,
            kernel_size=psf_parameters['kernel_size'],
            thickness=psf_parameters['thickness'],
            angle=psf_parameters['angle'],
            mode=psf_parameters['mode']
        )
    
    else:   # natural motion blur kernels
        filtered_image, kernel = apply_motion_blur_kernel(image)


    # Post-process the data and return np.array / tensor depends on the input
    if tensor_format:   # working with tensor input, returning a tensor output
        filtered_image = torchvision.transforms.ToTensor()(filtered_image)

    if return_kernel:
        return filtered_image, kernel
    return filtered_image


def debug():
    # Set sigma values for demonstration
    sigma_values = [0, 0.5, 1]
    for sigma in sigma_values:
        psf_parameters = generate_random_params(sigma)
        print("Generated PSF parameters:", psf_parameters)


def main():
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
            filtered_image, kernel = motion_blur(image, sigma=sigma, return_kernel=True)
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



# This ensures the main function (main block) will not run when motion_blur.py is imported
# If this file is imported then its name is *not* __main__, because it's not the main entry point, so the main() function won't be called
if __name__ == "__main__":
    main()
    #debug()
    #psf = np.zeros((50, 50, 3))
    #psf = cv2.ellipse(psf,
    #                  (25, 25),  # center
    #                  (22, 0),  # axes of the ellipse,  (blur length, PSF thickness)
    #                  90,  # angle of motion in degrees
    #                  0, 90,  # ful ellipse, not an arc
    #                  (1, 1, 1),  # white color
    #                  thickness=-1)  # filled

    #psf /= psf[:, :, 0].sum()  # normalize by sum of one channel
    # Convert to a single channel for display (optional)
    #psf_gray = psf[:, :, 0]  # Use just one channel
    #psf_gray = (psf_gray / psf_gray.max() * 255).astype(np.uint8)  # Normalize to 0-255

    # Show the kernel
    #cv2.imshow('Kernel', psf_gray)
    #cv2.waitKey(0)  # Wait for a key press to proceed
    #cv2.destroyAllWindows()