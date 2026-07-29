from __future__ import annotations
from ..common import *
from ..paths import *
from ..interfaces import *
from ..registry import *
from ..config import *

@register("preprocessor", "passthrough")
class PassthroughPreprocessor(Preprocessor):
    def process(self, frame: np.ndarray) -> PreprocessResult:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return PreprocessResult(bgr=frame, gray=gray, diagnostics={"enabled": False}, enhanced=False)


@register("preprocessor", "standard")
class StandardPreprocessor(Preprocessor):
    def __init__(self, frame_resize=(640, 480), blur_kernel=(5, 5), blur_sigma=0):
        self.frame_resize = tuple(frame_resize)
        self.blur_kernel = tuple(blur_kernel)
        self.blur_sigma = blur_sigma

    def process(self, frame: np.ndarray) -> PreprocessResult:
        bgr = cv2.resize(frame, self.frame_resize)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, self.blur_kernel, self.blur_sigma)
        return PreprocessResult(
            bgr=bgr, gray=gray,
            diagnostics={"blur_score": float(cv2.Laplacian(gray, cv2.CV_64F).var()), "brightness": float(gray.mean())},
            enhanced=False
        )


@register("preprocessor", "distortion_aware")
class DistortionAwarePreprocessor(Preprocessor):
    def __init__(
        self,
        frame_resize=(640, 480),
        blur_kernel=(5, 5),
        blur_sigma=0,
        blur_threshold=100.0,
        exposure_low=60.0,
        exposure_high=195.0,
        enhancement_for_blur="unsharp_mask",
        enhancement_for_exposure="clahe",
        unsharp_sigma=2.0,
        unsharp_strength=1.5,
        clahe_clip_limit=2.0,
        clahe_tile_grid=(8, 8),
        bilateral_d=9,
        bilateral_sigma_color=75,
        bilateral_sigma_space=75,
        use_multi_metric=True,
        gamma_correction=False,
        gamma_value=1.2,
        contrast_stretching=False,
    ):
        self.frame_resize = tuple(frame_resize)
        self.blur_kernel = tuple(blur_kernel)
        self.blur_sigma = blur_sigma
        self.blur_threshold = blur_threshold
        self.exposure_low = exposure_low
        self.exposure_high = exposure_high
        self.enhancement_for_blur = enhancement_for_blur
        self.enhancement_for_exposure = enhancement_for_exposure
        self.unsharp_sigma = unsharp_sigma
        self.unsharp_strength = unsharp_strength
        self.use_multi_metric = use_multi_metric
        self.gamma_correction = gamma_correction
        self.gamma_value = gamma_value
        self.contrast_stretching = contrast_stretching
        self.clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=tuple(clahe_tile_grid))
        self.bilateral_d = bilateral_d
        self.bilateral_sigma_color = bilateral_sigma_color
        self.bilateral_sigma_space = bilateral_sigma_space

    def _detect_blur_multi_metric(self, gray: np.ndarray) -> Dict[str, float]:
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
        tenengrad = np.sqrt(gx**2 + gy**2).mean()
        brenner = np.mean(np.abs(np.diff(gray.astype(np.float64), axis=1)))
        return {"laplacian_var": lap_var, "tenengrad": tenengrad, "brenner": brenner, "blur_score": 1.0 - min(lap_var / 500.0, 1.0)}

    def _gamma_correct(self, gray: np.ndarray) -> np.ndarray:
        inv_gamma = 1.0 / self.gamma_value
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype(np.uint8)
        return cv2.LUT(gray, table)

    def _contrast_stretch(self, gray: np.ndarray) -> np.ndarray:
        p2, p98 = np.percentile(gray, (2, 98))
        return cv2.normalize(gray, None, p2, p98, cv2.NORM_MINMAX)

    def process(self, frame: np.ndarray) -> PreprocessResult:
        bgr = cv2.resize(frame, self.frame_resize)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        if self.use_multi_metric:
            blur_metrics = self._detect_blur_multi_metric(gray)
            blur_score = blur_metrics["laplacian_var"]
        else:
            blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        brightness = float(gray.mean())

        is_blurry = blur_score < self.blur_threshold
        is_underexposed = brightness < self.exposure_low
        is_overexposed = brightness > self.exposure_high

        if is_blurry and (is_underexposed or is_overexposed):
            distortion_type = "blurry+exposure"
        elif is_blurry:
            distortion_type = "blurry"
        elif is_underexposed:
            distortion_type = "underexposed"
        elif is_overexposed:
            distortion_type = "overexposed"
        else:
            distortion_type = "pristine"

        enhanced = False

        if self.gamma_correction and (is_underexposed or is_overexposed):
            gray = self._gamma_correct(gray)
            enhanced = True

        if is_blurry:
            if self.enhancement_for_blur == "unsharp_mask":
                gray = self._unsharp_mask(gray)
                enhanced = True
            elif self.enhancement_for_blur == "wiener":
                gray = self._wiener_filter(gray)
                enhanced = True

        if is_underexposed or is_overexposed:
            if self.enhancement_for_exposure == "clahe":
                gray = self.clahe.apply(gray)
                enhanced = True
            elif self.enhancement_for_exposure == "bilateral":
                gray = cv2.bilateralFilter(gray, self.bilateral_d, self.bilateral_sigma_color, self.bilateral_sigma_space)
                enhanced = True

        if self.contrast_stretching and (is_underexposed or is_overexposed):
            gray = self._contrast_stretch(gray)
            enhanced = True

        gray = cv2.GaussianBlur(gray, self.blur_kernel, self.blur_sigma)

        diagnostics = {
            "blur_score": blur_score,
            "brightness": brightness,
            "distortion_type": distortion_type,
            "enhanced": enhanced,
            "is_blurry": is_blurry,
            "is_underexposed": is_underexposed,
            "is_overexposed": is_overexposed,
        }
        if self.use_multi_metric:
            diagnostics.update(blur_metrics)

        return PreprocessResult(bgr=bgr, gray=gray, enhanced=enhanced, diagnostics=diagnostics)

    def _unsharp_mask(self, gray: np.ndarray) -> np.ndarray:
        blurred = cv2.GaussianBlur(gray, (0, 0), self.unsharp_sigma)
        local_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        adaptive_strength = self.unsharp_strength * (1.0 + np.tanh(local_var / 100.0))
        return cv2.addWeighted(gray, 1.0 + adaptive_strength, blurred, -adaptive_strength, 0)

    def _wiener_filter(self, gray: np.ndarray) -> np.ndarray:
        kernel = (5, 5)
        f64 = gray.astype(np.float64)
        mean = cv2.blur(f64, kernel)
        mean_sq = cv2.blur(f64 ** 2, kernel)
        variance = mean_sq - mean ** 2
        noise_mask = variance < np.percentile(variance, 20)
        noise_var = np.mean(variance[noise_mask]) if noise_mask.any() else 100.0
        ratio = np.maximum(0, variance - noise_var) / np.maximum(variance, noise_var)
        result = mean + ratio * (f64 - mean)
        return np.clip(result, 0, 255).astype(np.uint8)
