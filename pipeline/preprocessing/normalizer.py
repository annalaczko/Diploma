from pathlib import Path

from PIL import Image
from tqdm import tqdm

from pipeline.preprocessing.preprocessing_step import PreprocessingStep
from pipeline.preprocessing.dataset import collect_image_paths


class Normalizer(PreprocessingStep):
    """Pad images to a square canvas, resize them to 256x256, and save them.

    This gives every downstream step (feature learning, YOLO, ...) images
    with a consistent size and aspect ratio, without distorting the
    original content: padding preserves the aspect ratio, only the final
    resize changes scale.

    Supports a single file, a directory, or a list/tuple of valid image paths.
    Missing paths are ignored. The output directory is returned by run().
    """

    def __init__(
        self,
        input_path=None,
        output_path=None,
        name=None,
        delete_input_data=False,
        run_id="default_run",
        target_size: int = 256,
        background_color=(0, 0, 0),
        image_extensions: tuple[str, ...] = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"),
        rename=False,
        **kwargs,
    ):
        """
        Args:
            input_path: a single file, a directory, or a list/tuple of
                either. Directories are searched recursively.
            output_path: where normalized images are written. Defaults to
                the Step-computed run-scoped directory if not given.
            target_size: side length (px) of the final square image.
            background_color: RGB fill color used for the padding added
                around non-square source images.
            image_extensions: file suffixes treated as input images.
            rename: currently unused/reserved; output files keep their
                original filenames regardless of this flag.
        """
        super().__init__(
            input_path=input_path,
            output_path=output_path,
            name=name,
            delete_input_data=delete_input_data,
            run_id=run_id,
            **kwargs,
        )
        self.target_size = target_size
        self.background_color = background_color
        self.image_extensions = image_extensions
        self.rename = rename

    def run(self) -> Path:
        """Pad each image to a square, resize to 256x256, and save under output_path."""
        output_dir = Path(self.output_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        image_paths = collect_image_paths(self.input_path, self.image_extensions)
        if not image_paths:
            raise FileNotFoundError(
                f"No existing images with extensions {self.image_extensions} found under {self.input_path}"
            )

        for image_path in tqdm(image_paths, desc="Normalizing images", unit="image"):
            image = Image.open(image_path).convert("RGB")
            padded = self.pad_to_square(image, fill_color=self.background_color)
            resized = padded.resize((self.target_size, self.target_size), Image.Resampling.LANCZOS)
            resized.save(output_dir / image_path.name)

        if self.delete_input_data_flag:
            self.delete_input_data()

        return output_dir

    def pad_to_square(self, image: Image.Image, fill_color=(0, 0, 0)) -> Image.Image:
        """Pad the image with a background color to the nearest square size.

        Centers the original image on a new `max(width, height)`-sized
        canvas filled with `fill_color`, so aspect ratio is preserved
        (unlike a plain resize to a square, which would stretch it).
        """
        width, height = image.size
        max_side = max(width, height)

        padded = Image.new("RGB", (max_side, max_side), fill_color)
        offset_x = (max_side - width) // 2
        offset_y = (max_side - height) // 2
        padded.paste(image, (offset_x, offset_y))
        return padded
