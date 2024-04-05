This is Implementing a CycleGAN model for image translation. 


<h3>Generate MRI images of different contrast levels using cyclic GANs</h3>
<br>
**Capstone Summary:** Medical misdiagnosis is an extremely serious problem, but it also happens far too frequently. Since interpreting imaging methods in the medical profession is not a straightforward binary process (Normal or Abnormal), it is necessary to have a radiologist's opinion. Even so, it's possible for one radiologist to detect something another misses. Conflicting information might result from this, making it challenging to appropriately suggest therapy options to the patient.


The diagnosis of an MRI (Magnetic Resonance Imaging) is one of the challenging challenges in medical imaging. The radiologist may occasionally require several imaging modalities to interpret the scan, which can significantly improve diagnosis accuracy by giving practitioners a more complete knowledge.

But getting access to various imaging is expensive and complex. We can employ style transfer to create artificial MRI images of various contrast levels from pre-existing MRI scans with the use of deep learning. With the use of a second image, this will aid in providing a better diagnosis.

To transfer the style of one MRI image to another in this capstone and improve understanding of the scanned image, CycleGAN will be used. You can convert T1 weighted MRI scans into T2 weighted pictures using GANs, and vice versa.
<br>
**Project Statement:** To develop a generative adversarial model (modified U-Net) that can create synthetic MRI pictures with various contrast levels from existing MRI scans

**Dataset Used:** MRI+T1_T2+Dataset

**NOTE:** Since this is an unpaired dataset, there is no relationship whatsoever between the T1 and T2 MRI images that are part of it.


**Observations:**

* The number of images in the dataset is less compared to usual training datasets. We can leverage the ability of Generative Adversarial Networks to generate images of unpaired images by training for more epochs and potentially using less data augmentation.
* The dataset consists of only 89 images with dimensions 217*181.
* The images are in two different folders (t1 and t2) and have separate styles. They are also not paired.
* The images are resized to (64*64*1) for input as it's easier to process and less time consuming compared to higher dimensions. Using (128*128*1) resulted in an Out Of Memory Error due to insufficient VRAM on the GTX 1060 6GB GPU.
* A batch of 32 images is used from a total of 43 images in the T1 folder and 32 from a total of 46 images in the T2 folder.
* The generator and discriminator have millions of parameters to train.


**Note:** The images on the left have a particular style, and the images on the right have a different style. The generator is trained to adapt to these styles and mimic them.


Here's a breakdown of the code:

**Imports:**

- Necessary libraries for data science (NumPy, matplotlib), deep learning (TensorFlow), image processing (Skimage, imageio, PIL), and system navigation (OS, glob, sys) are imported.
- A function `is_imported` checks if a module is present in the environment.

**Path Creation:**

- The script defines paths for the training data folders containing T1 and T2 MRI images based on the current platform (Linux, macOS, or Windows).

**Image Loading and Preprocessing:**

- A function `image_extractor` takes a folder path and a list of image names and reads them using `imageio.imread`. 
- The `image_data` function displays a subplot of images in a given folder.
- The `folder_images_to_ndarray` function converts all images in a folder to a NumPy array.
- Several functions are defined for data normalization (`normalize`), resizing (`img_resize`), conversion between data types (`uint8_to_float32`, `float32_to_uint8`), and data shuffling (`shuffle_batch_data`).

**Main Training Script:**

- The `preprocess_image_train` function performs the following steps on the training data:
    - Converts images to NumPy arrays.
    - Converts data type to float32 for calculations.
    - Normalizes pixel values to the range [-1, 1].
    - Resizes images to 64x64.
    - Reshapes the data for the model.
    - Applies random left-right flips for data augmentation.
    - Shuffles the data using a batch size.
- The script processes both T1 and T2 data with a batch size of 32 and stores them in separate variables (`t1_processed` and `t2_processed`).
- Sample images from each processed dataset are visualized using `plt.imshow`.

**U-Net Generator and Discriminator Architecture:**

- The `InstanceNormalization` class implements instance normalization for image data.
- The `downsample` function defines a convolutional layer with LeakyReLU activation for downsampling.
- The `upsample` function defines a transposed convolutional layer with ReLU activation for upsampling.
- The `unet_generator` function defines the U-Net generator architecture with a combination of downsampling, upsampling, and skip connections. It uses the `InstanceNormalization` class and tanh activation in the final layer.
- The `discriminator` function defines the PatchGAN discriminator architecture with downsampling layers using `InstanceNormalization` and LeakyReLU activation.

**Model Initialization:**

- Two U-Net generator models (`generator_g` and `generator_f`) are created for translating between T1 and T2 images.
- Two PatchGAN discriminator models (`discriminator_x` and `discriminator_y`) are created for discriminating between real and generated images.

**Generator Output Visualization:**

- Sample noise is passed through the generators to verify their initial untrained output.
- The generated images are visualized using `plt.imshow`.

**Loss Functions and Training (not implemented in the provided code):**

- The script defines a Lambda value (`LAMBDA`) for weighting the cycle-consistency loss.
- Loss functions for the generator and discriminator models are typically defined here (e.g., mean squared error for image reconstruction, binary cross-entropy for discriminator).
- An optimizer (e.g., Adam) is chosen to update model weights during training.
- A training loop iterates over the processed data, performs forward passes through the generators and discriminators, calculates losses, and updates model weights using the optimizer.

**Overall, the provided code implements the core components of a CycleGAN model for image translation. It defines the U-Net generator and PatchGAN discriminator architectures, performs data preprocessing, and visualizes the initial generator output. The missing part is the training loop where the models are trained on the prepared data.**

