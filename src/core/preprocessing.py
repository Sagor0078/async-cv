import cv2
import numpy as np
from typing import Tuple, Optional, List
from PIL import Image, ImageEnhance
import io


class ImagePreprocessor:
    """Image preprocessing utilities for ML models"""

    @staticmethod
    def resize_image(
        image: np.ndarray,
        target_size: Tuple[int, int],
        maintain_aspect_ratio: bool = True,
    ) -> np.ndarray:
        """
        Resize image to target size

        Args:
            image: Input image as numpy array
            target_size: Target (width, height)
            maintain_aspect_ratio: Whether to maintain aspect ratio

        Returns:
            Resized image
        """
        if maintain_aspect_ratio:
            h, w = image.shape[:2]
            target_w, target_h = target_size

            # Calculate scaling factor
            scale = min(target_w / w, target_h / h)
            new_w, new_h = int(w * scale), int(h * scale)

            # Resize image
            resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

            # Create canvas and center the image
            canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
            y_offset = (target_h - new_h) // 2
            x_offset = (target_w - new_w) // 2
            canvas[y_offset : y_offset + new_h, x_offset : x_offset + new_w] = resized

            return canvas
        else:
            return cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)

    @staticmethod
    def normalize_image(
        image: np.ndarray,
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    ) -> np.ndarray:
        """
        Normalize image with given mean and std (ImageNet defaults)

        Args:
            image: Input image (0-255 range)
            mean: RGB mean values
            std: RGB std values

        Returns:
            Normalized image
        """
        # Convert to float and normalize to [0, 1]
        image = image.astype(np.float32) / 255.0

        # Apply normalization
        for i in range(3):
            image[:, :, i] = (image[:, :, i] - mean[i]) / std[i]

        return image

    @staticmethod
    def enhance_image(
        image_bytes: bytes,
        brightness: float = 1.0,
        contrast: float = 1.0,
        saturation: float = 1.0,
        sharpness: float = 1.0,
    ) -> bytes:
        """
        Enhance image quality

        Args:
            image_bytes: Input image as bytes
            brightness: Brightness factor (1.0 = no change)
            contrast: Contrast factor (1.0 = no change)
            saturation: Saturation factor (1.0 = no change)
            sharpness: Sharpness factor (1.0 = no change)

        Returns:
            Enhanced image as bytes
        """
        # Load image
        image = Image.open(io.BytesIO(image_bytes))

        # Apply enhancements
        if brightness != 1.0:
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(brightness)

        if contrast != 1.0:
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(contrast)

        if saturation != 1.0:
            enhancer = ImageEnhance.Color(image)
            image = enhancer.enhance(saturation)

        if sharpness != 1.0:
            enhancer = ImageEnhance.Sharpness(image)
            image = enhancer.enhance(sharpness)

        # Convert back to bytes
        output = io.BytesIO()
        image.save(output, format="JPEG", quality=95)
        return output.getvalue()

    @staticmethod
    def denoise_image(image: np.ndarray, method: str = "bilateral") -> np.ndarray:
        """
        Apply denoising to image

        Args:
            image: Input image
            method: Denoising method ('bilateral', 'gaussian', 'median')

        Returns:
            Denoised image
        """
        if method == "bilateral":
            return cv2.bilateralFilter(image, 9, 75, 75)
        elif method == "gaussian":
            return cv2.GaussianBlur(image, (5, 5), 0)
        elif method == "median":
            return cv2.medianBlur(image, 5)
        else:
            return image

    @staticmethod
    def adjust_gamma(image: np.ndarray, gamma: float = 1.0) -> np.ndarray:
        """
        Apply gamma correction

        Args:
            image: Input image
            gamma: Gamma value (< 1 = brighter, > 1 = darker)

        Returns:
            Gamma corrected image
        """
        inv_gamma = 1.0 / gamma
        table = np.array(
            [((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]
        ).astype("uint8")
        return cv2.LUT(image, table)

    @staticmethod
    def convert_colorspace(
        image: np.ndarray, source: str = "BGR", target: str = "RGB"
    ) -> np.ndarray:
        """
        Convert between color spaces

        Args:
            image: Input image
            source: Source color space
            target: Target color space

        Returns:
            Converted image
        """
        conversion_map = {
            ("BGR", "RGB"): cv2.COLOR_BGR2RGB,
            ("RGB", "BGR"): cv2.COLOR_RGB2BGR,
            ("BGR", "GRAY"): cv2.COLOR_BGR2GRAY,
            ("RGB", "GRAY"): cv2.COLOR_RGB2GRAY,
            ("BGR", "HSV"): cv2.COLOR_BGR2HSV,
            ("RGB", "HSV"): cv2.COLOR_RGB2HSV,
        }

        conversion_code = conversion_map.get((source, target))
        if conversion_code:
            return cv2.cvtColor(image, conversion_code)
        else:
            return image

    @staticmethod
    def crop_center(image: np.ndarray, crop_size: Tuple[int, int]) -> np.ndarray:
        """
        Crop image from center

        Args:
            image: Input image
            crop_size: (width, height) to crop

        Returns:
            Cropped image
        """
        h, w = image.shape[:2]
        crop_w, crop_h = crop_size

        start_x = max(0, (w - crop_w) // 2)
        start_y = max(0, (h - crop_h) // 2)

        return image[start_y : start_y + crop_h, start_x : start_x + crop_w]

    @staticmethod
    def pad_image(
        image: np.ndarray, target_size: Tuple[int, int], pad_value: int = 0
    ) -> np.ndarray:
        """
        Pad image to target size

        Args:
            image: Input image
            target_size: Target (width, height)
            pad_value: Value to use for padding

        Returns:
            Padded image
        """
        h, w = image.shape[:2]
        target_w, target_h = target_size

        if w >= target_w and h >= target_h:
            return image

        # Calculate padding
        pad_w = max(0, target_w - w)
        pad_h = max(0, target_h - h)

        # Pad image
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left

        return cv2.copyMakeBorder(
            image,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            cv2.BORDER_CONSTANT,
            value=[pad_value] * 3,
        )

    @staticmethod
    def augment_image(
        image: np.ndarray,
        rotation_angle: Optional[float] = None,
        flip_horizontal: bool = False,
        flip_vertical: bool = False,
        scale_factor: Optional[float] = None,
    ) -> np.ndarray:
        """
        Apply data augmentation transformations

        Args:
            image: Input image
            rotation_angle: Rotation angle in degrees
            flip_horizontal: Whether to flip horizontally
            flip_vertical: Whether to flip vertically
            scale_factor: Scale factor for resizing

        Returns:
            Augmented image
        """
        h, w = image.shape[:2]

        # Rotation
        if rotation_angle is not None:
            center = (w // 2, h // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
            image = cv2.warpAffine(image, rotation_matrix, (w, h))

        # Flipping
        if flip_horizontal:
            image = cv2.flip(image, 1)

        if flip_vertical:
            image = cv2.flip(image, 0)

        # Scaling
        if scale_factor is not None and scale_factor != 1.0:
            new_w, new_h = int(w * scale_factor), int(h * scale_factor)
            image = cv2.resize(image, (new_w, new_h))

        return image


class BatchPreprocessor:
    """Batch preprocessing utilities"""

    def __init__(self, target_size: Tuple[int, int] = (224, 224)):
        self.target_size = target_size
        self.preprocessor = ImagePreprocessor()

    def preprocess_batch(
        self, images: List[np.ndarray], normalize: bool = True, enhance: bool = False
    ) -> np.ndarray:
        """
        Preprocess batch of images

        Args:
            images: List of images
            normalize: Whether to normalize images
            enhance: Whether to apply enhancement

        Returns:
            Batch tensor of preprocessed images
        """
        batch = []

        for image in images:
            # Resize
            processed = self.preprocessor.resize_image(image, self.target_size)

            # Enhance if requested
            if enhance:
                processed = self.preprocessor.denoise_image(processed)
                processed = self.preprocessor.adjust_gamma(processed, 1.2)

            # Normalize if requested
            if normalize:
                processed = self.preprocessor.normalize_image(processed)

            batch.append(processed)

        return np.array(batch)
