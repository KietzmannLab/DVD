import os
import re
import cv2
import json
import h5py
import time
import math
import subprocess
import wandb
import argparse
import random
import numpy as np
import scipy.stats as stats
from collections import Counter
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter, ImageChops

import os
import torch
import torchvision
import timm
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from collections import defaultdict
from torchvision import transforms
import torchvision.transforms.functional as F
import torch.optim.lr_scheduler as lr_scheduler
from torch.optim.lr_scheduler import _LRScheduler
from torchvision.datasets import ImageFolder

import torch
import torch.nn as nn
import kornia.augmentation as K
import kornia.filters as KF

##########################
## Visual acuity & Color & Contrast Sensitivity Development Strategy
##########################
class EarlyVisualDevelopmentTransformer:

    def __init__(self):
        pass

    def get_early_visual_acuity(self, age_months):
        """Returns the visual acuity in Gaussian blur sigma for a given age in months.

           Reference: 
            Braddick, O., & Atkinson, J. (2011). Development of human visual function. Vision Research, 51(13), 1588–1609. https://doi.org/10.1016/j.visres.2011.02.018
            Caltrider, D., Gupta, A., & Tripathy, K. (2024). Evaluation of Visual Acuity. In StatPearls. StatPearls Publishing. http://www.ncbi.nlm.nih.gov/books/NBK564307/
            Dobson, V., Clifford-Donaldson, C. E., Green, T. K., Miller, J. M., & Harvey, E. M. (2009). Normative Monocular Visual Acuity for Early Treatment Diabetic Retinopathy Study Charts in Emmetropic Children 5 to 12 Years of Age. Ophthalmology, 116(7), 1397–1401. https://doi.org/10.1016/j.ophtha.2009.01.019
            Drover, J. R., Cornick, S. L., Hallett, D., Drover, A., Mayo, D., & Kielly, N. (2017). Normative pediatric data for three tests of functional vision. Canadian Journal of Ophthalmology, 52(2), 198–202. https://doi.org/10.1016/j.jcjo.2016.08.016
            Drover, J. R., Felius, J., Cheng, C. S., Morale, S. E., Wyatt, L., & Birch, E. E. (2008). Normative pediatric visual acuity using single surrounded HOTV optotypes on the Electronic Visual Acuity Tester following the Amblyopia Treatment Study protocol. Journal of American Association for Pediatric Ophthalmology and Strabismus, 12(2), 145–149. https://doi.org/10.1016/j.jaapos.2007.08.014
            El-Gohary, A., Abuelela, M., & Eldin, A. (2017). Age norms for grating acuity and contrast sensitivity measured by Lea tests in the first three years of life. International Journal of Ophthalmology, 10, 1150–1153. https://doi.org/10.18240/ijo.2017.07.20
            Gal Katzhendler, Katzhendler, G., Daphna Weinshall, & Weinshall, D. (2019). Potential upside of high initial visual acuity. Proceedings of the National Academy of Sciences of the United States of America, 116(38), 18765–18766. https://doi.org/10.1073/pnas.1906400116
            Inal, A., Ocak, O. B., Aygit, E. D., Yilmaz, I., Inal, B., Taskapili, M., & Gokyigit, B. (2018). Comparison of visual acuity measurements via three different methods in preschool children: Lea symbols, crowded Lea symbols, Snellen E chart. International Ophthalmology, 38(4), 1385–1391. https://doi.org/10.1007/s10792-017-0596-1
            Katzhendler, G., & Weinshall, D. (2019). Blurred Images Lead to Bad Local Minima (arXiv:1901.10788). arXiv. http://arxiv.org/abs/1901.10788
            Kugelberg, U. (1992a). Visual acuity following treatment of bilateral congenital cataracts. Documenta Ophthalmologica, 82(3), 211–215. https://doi.org/10.1007/BF00160767
            Leone, J. F., Mitchell, P., Kifley, A., Rose, K. A., & Sydney Childhood Eye Studies. (2014). Normative visual acuity in infants and preschool‐aged children in S ydney. Acta Ophthalmologica, 92(7). https://doi.org/10.1111/aos.12366
            Lukas Vogelsang, Lukas Vogelsang, Vogelsang, L., Sharon Gilad-Gutnick, Gilad-Gutnick, S., Evan Ehrenberg, Ehrenberg, E., Albert Yonas, Yonas, A., Sidney Diamond, Diamond, S., Richard Held, Held, R., Pawan Sinha, & Sinha, P. (2018). Potential downside of high initial visual acuity. Proceedings of the National Academy of Sciences of the United States of America, 115(44), 11333–11338. https://doi.org/10.1073/pnas.1800901115
            Lukas Vogelsang, Vogelsang, L., Lukas Vogelsang, Lukas Vogelsang, Lukas Vogelsang, Sharon Gilad-Gutnick, Gilad-Gutnick, S., Diamond, S., Sidney Diamond, Diamond, S., Albert Yonas, Yonas, A., Pawan Sinha, & Sinha, P. (2019). Response to Katzhendler and Weinshall: Initial visual degradation during development may be adaptive. Proceedings of the National Academy of Sciences of the United States of America, 116(38), 18767–18768. https://doi.org/10.1073/pnas.1910674116
            Morale, S. E., Cheng-Patel, C. S., Jost, R. M., Donohoe, N., Leske, D. A., & Birch, E. E. (2021). Normative pediatric visual acuity using electronic early treatment for diabetic retinopathy protocol. Journal of AAPOS : The Official Publication of the American Association for Pediatric Ophthalmology and Strabismus, 25(3), 172–175. https://doi.org/10.1016/j.jaapos.2021.01.003
            Nolan, J. M., Power, R., Stringham, J., Dennison, J., Stack, J., Kelly, D., Moran, R., Akuffo, K. O., Corcoran, L., & Beatty, S. (2016). Enrichment of Macular Pigment Enhances Contrast Sensitivity in Subjects Free of Retinal Disease: Central Retinal Enrichment Supplementation Trials – Report 1. Investigative Ophthalmology & Visual Science, 57(7), 3429–3439. https://doi.org/10.1167/iovs.16-19520
            Norcia, A. M., Tyler, C. W., & Hamer, R. D. (1990). Development of contrast sensitivity in the human infant. Vision Research, 30(10), 1475–1486. https://doi.org/10.1016/0042-6989(90)90028-J
            O’Connor, A. R., & Milling, A. (2020). Normative data for the redesigned Kay Pictures visual acuity test. Journal of American Association for Pediatric Ophthalmology and Strabismus, 24(4), 242–244. https://doi.org/10.1016/j.jaapos.2020.05.003
            Pan, Y., Tarczy-Hornoch, K., Susan A., C., Wen, G., Borchert, M. S., Azen, S. P., & Varma, R. (2009). Visual Acuity Norms in Preschool Children: The Multi-Ethnic Pediatric Eye Disease Study. Optometry and Vision Science : Official Publication of the American Academy of Optometry, 86(6), 607–612. https://doi.org/10.1097/OPX.0b013e3181a76e55
            Potential upside of high initial visual acuity? (n.d.). https://doi.org/10.1073/pnas.1906400116
            Sanker, N., Dhirani, S., & Bhakat, P. (2013). Comparison of visual acuity results in preschool children with lea symbols and bailey-lovie e chart. Middle East African Journal of Ophthalmology, 20(4), 345. https://doi.org/10.4103/0974-9233.120020
            The Newborn Senses: Sight and Eye Color. (n.d.). Lozier Institute. Retrieved May 31, 2024, from https://lozierinstitute.org/dive-deeper/the-newborn-senses-sight-and-eye-color/
        """

        a, b, c, d = 18.035945052640425, 0.7933899743217134, 1.6012490927401029, 0.027054604551482078
        return a * np.exp(-b * age_months) + c * np.exp(-d * age_months)

    def get_color_sensitivity(self, age_months): 
        """Calculates the color factor using a custom function.

            Reference: 
             Knoblauch, K., Vital-Durand, F., & Barbur, J. L. (2001). Variation of chromatic sensitivity across the life span. Vision Research, 41(1), 23-36. doi:10.1016/S0042-6989(00)00205-4 
             Crognale, Weiss, Kelly & Teller, 1998, Vision Research doi.org/10.1016/S0042-6989(98)00074-1 

        """
        color_params = {
            "AverageRGBDevelop": {
                "a": 0.008604133954779169,
                "b": 4.380740053287391e-05,
                "alpha": 0.8807610802743646
            },
            'min_sensitivity_threshold_ages': (21.2 + 21.1 + 18.8) / 3,
        }

        def T(age, a, b, alpha):
            return a * age ** (-alpha) + b * age ** alpha

        def custom_average_color_mix(age_months):
            age_years = age_months / 12  # Convert months to years
            param = color_params["AverageRGBDevelop"]
            a, b, alpha = param["a"], param["b"], param["alpha"]
            min_age = color_params["min_sensitivity_threshold_ages"]
            return (T(min_age, a, b, alpha) / T(age_years, a, b, alpha)) if age_years != 0 else 0

        return custom_average_color_mix(age_months)


    def get_contrast_sensitivity_development(self, age_months, age50=4.8*12, n=2.1633375920569247):
        """Reference:
        
        Braddick, O., & Atkinson, J. (2011). Development of human visual function. Vision Research, 51(13), 1588–1609. https://doi.org/10.1016/j.visres.2011.02.018
        Brown, A. M., Lindsey, D. T., Cammenga, J. G., Giannone, P. J., & Stenger, M. R. (2015). The Contrast Sensitivity of the Newborn Human Infant. Investigative Ophthalmology & Visual Science, 56(1), 625–632. https://doi.org/10.1167/iovs.14-14757
        Dekker, T. M., Farahbakhsh, M., Atkinson, J., Braddick, O. J., & Jones, P. R. (2020). Development of the spatial contrast sensitivity function (CSF) during childhood: Analysis of previous findings and new psychophysical data. Journal of Vision, 20(13), 4. https://doi.org/10.1167/jov.20.13.4
        El-Gohary, A., Abuelela, M., & Eldin, A. (2017). Age norms for grating acuity and contrast sensitivity measured by Lea tests in the first three years of life. International Journal of Ophthalmology, 10, 1150–1153. https://doi.org/10.18240/ijo.2017.07.20
        Nolan, J. M., Power, R., Stringham, J., Dennison, J., Stack, J., Kelly, D., Moran, R., Akuffo, K. O., Corcoran, L., & Beatty, S. (2016). Enrichment of Macular Pigment Enhances Contrast Sensitivity in Subjects Free of Retinal Disease: Central Retinal Enrichment Supplementation Trials – Report 1. Investigative Ophthalmology & Visual Science, 57(7), 3429–3439. https://doi.org/10.1167/iovs.16-19520
        Norcia, A. M., Tyler, C. W., & Hamer, R. D. (1990). Development of contrast sensitivity in the human infant. Vision Research, 30(10), 1475–1486. https://doi.org/10.1016/0042-6989(90)90028-J
        Object,  object. (n.d.). Using a single test to measure human contrast sensitivity from early childhood to maturity. Retrieved July 12, 2024, from https://core.ac.uk/reader/82492300
        Owsley, C., Sekuler, R., & Siemsen, D. (1983). Contrast sensitivity throughout adulthood. Vision Research, 23(7), 689–699. https://doi.org/10.1016/0042-6989(83)90210-9
        Pateras, E., & Karioti, M. (2020). Contrast Sensitivity Studies and Test- A Review. International Journal of Ophthalmology and Clinical Research, 7(2). https://doi.org/10.23937/2378-346X/1410116
        Pelli, D. G., & Bex, P. (2013). Measuring contrast sensitivity. Vision Research, 90, 10–14. https://doi.org/10.1016/j.visres.2013.04.015
        Stavros, K. A., & Kiorpes, L. (2008). Behavioral measurement of temporal contrast sensitivity development in macaque monkeys (Macaca nemestrina). Vision Research, 48(11), 1335–1344. https://doi.org/10.1016/j.visres.2008.01.031
        Teller, D. Y. (1998). Spatial and temporal aspects of infant color vision. Vision Research, 38(21), 3275–3282. https://doi.org/10.1016/S0042-6989(97)00468-9

        """
        y_max = (300**n) /(300 ** n + age50 ** n) # when age_months = 300 is max | 25 years old
        return (age_months ** n) / (age_months ** n + age50 ** n) / y_max  # Range in [0,1]


    def apply_fft_transformations(self, image, age_months, apply_blur=1, apply_color=1, apply_contrast=1, contrast_threshold =0.2, image_size= 224, verbose=False):
        """Applies a contrast sensitivity filter to the image based on the age of the subject.

        Args:
            image (PIL.Image or Tensor): The input image to be processed.
            age_months (int): The age in months which determines the visual filter settings.
            verbose (bool): If True, prints additional debug information.
        
        Returns:
            PIL.Image: The filtered image.
        """
        def process_image(image):
            if verbose:
                print(f"Original mean: {image.mean()} and {image.std}")

            # Visual acuity
            if apply_blur:
                # Compute blur_sigma based on the age in months
                blur_sigma = self.get_early_visual_acuity(age_months) * ( image_size / 224 ) # Sigma scaling with image size 
                if blur_sigma > 0:
                    kernel_size = int(8 * blur_sigma) + (1 if int(8 * blur_sigma) % 2 == 0 else 0)
                    image = KF.gaussian_blur2d(
                        image, (kernel_size, kernel_size),
                        (blur_sigma, blur_sigma),
                        border_type="reflect"
                    )

            # Color
            if apply_color:
                color_factor = self.get_color_sensitivity(age_months)
                image = self.interpolate_color_grayscale(image, color_factor)

            # Contrast
            if apply_contrast:
                # Compute contrast sensitivity with a small offset to avoid division by zero
                contrast_sensitivity = self.get_contrast_sensitivity_development(age_months) + 1e-10
                # Process each channel (R, G, B) in the frequency domain
                fft_channels = [torch.fft.fft2(image[:, i, :, :]) for i in range(3)]  # List of [Batch_size, H, W] tensors
                # Compute the power spectrum for each channel
                power_spectra = [torch.abs(fft_channel) ** 2 for fft_channel in fft_channels]
                # Set a dynamic threshold based on age and contrast sensitivity
                max_power = max([power_spectra[i].max() for i in range(3)]) # RGB  #* max of all channels
                threshold = max_power* (1 - contrast_sensitivity) * 0.001 * contrast_threshold # contrast_threshold higher lower,  to control the speed of contrast drop speed, since it filters out the high frequency more
                # Apply thresholding to suppress low-power frequencies
                fft_filtered = [fft_channel * (power_spectrum >= threshold)
                                for fft_channel, power_spectrum in zip(fft_channels, power_spectra)]

                # Perform inverse FFT to obtain the filtered image in the spatial domain
                filtered_channels = [torch.fft.ifft2(fft_channel).real for fft_channel in fft_filtered]
                # Stack the filtered channels back into a single tensor
                image = torch.stack(filtered_channels, dim=1)  # Shape: [batch, 3, H, W]
                # Ensure pixel values are within the valid range [0, 1]
                image = image.clip(0, 1)

            return image

        # If the input is a list, process each image individually
        if isinstance(image, list):
            return [process_image(img) for img in image]
        else:
            return process_image(image)
            
    @staticmethod
    def interpolate_color_grayscale(image, color_factor):
        # Convert the image to grayscale
        grayscale_image = F.rgb_to_grayscale(image, num_output_channels=3)
        
        # Blend the original image and the grayscale image based on the color_factor
        blended_image = color_factor * image + (1 - color_factor) * grayscale_image
        return blended_image


#* Determine the development time order: normal, random, mid-phase ...
def generate_age_months_curve(total_epochs, len_train_loader, months_per_epoch, shuffle=False, seed=None, mid_phase=False):
    """
    Generate the sequence of age_months based on the epochs and the number of batches.
    
    Args:
    - total_epochs (int): Total number of epochs.
    - len_train_loader (int): Number of batches in the training loader.
    - months_per_epoch (float): Months per epoch.
    - shuffle (bool): If True, shuffle the age_months curve.
    - seed (int, optional): Seed for reproducibility when shuffling.
    - mid_phase (bool): If True, the ages first go down in an alternating manner, then go up in an alternating manner.

    Returns:
    - age_months_curve (list): The generated (and possibly shuffled or mid-phased) age months curve.
    """
    age_months_curve = []
    
    for epoch in range(0, total_epochs):
        for batch_id in range(len_train_loader):
            age_month = (epoch - 0) * months_per_epoch + batch_id * months_per_epoch / len_train_loader
            age_months_curve.append(age_month)

    if mid_phase:
        half = len(age_months_curve) // 2
        
        # Interleave the descending and ascending curves: first half decrease, second half increase
        mid_phase_age_months_curve = [None] * len(age_months_curve)
        mid_phase_age_months_curve[:half] = sorted(age_months_curve[::2], reverse=True)
        mid_phase_age_months_curve[half:] = age_months_curve[1::2]
        assert len(mid_phase_age_months_curve) ==  len(age_months_curve)
        return mid_phase_age_months_curve

    if shuffle:
        if seed is not None:
            random.seed(seed)
        random.shuffle(age_months_curve)
        return age_months_curve
    
    return age_months_curve