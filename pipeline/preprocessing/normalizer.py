"""Normalizer: pads and resizes every image to one square size."""

from pathlib import Path

from PIL import Image
from tqdm import tqdm

from pipeline.preprocessing.preprocessing_step import PreprocessingStep
from pipeline.preprocessing.dataset import ImageRecord, collect_image_paths


class Normalizer(PreprocessingStep):
    """Make every image the same size: pad it to a square (centered, with
    `background_color` around it), then resize it to `target_size` x
    `target_size` (256 by default).

    Padding instead of stretching keeps the drawing's proportions; only the
    final resize changes its scale. The feature learners need one fixed
    input size, so a Normalizer is part of every preprocessing chain.
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
            rename: unused (output files keep their names); kept because
                it is part of the step's logged settings.
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
        """Disk mode: normalize every image under `input_path` and save it
        under the same name in `output_path`."""
        output_dir = Path(self.output_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        image_paths = collect_image_paths(self.input_path, self.image_extensions)
        if not image_paths:
            raise FileNotFoundError(
                f"No existing images with extensions {self.image_extensions} found under {self.input_path}"
            )

        for image_path in tqdm(image_paths, desc="Normalizing images", unit="image"):
            self.normalize(Image.open(image_path)).save(output_dir / image_path.name)

        if self.delete_input_data_flag:
            self.delete_input_data()

        return output_dir

    def process_batch(self, records: list[ImageRecord]) -> list[ImageRecord]:
        """In-memory mode: normalize each image, keeping its name."""
        return [ImageRecord(r.name, self.normalize(r.image)) for r in records]

    def normalize(self, image: Image.Image) -> Image.Image:
        """Pad one image to a square and resize it to `target_size`."""
        padded = self.pad_to_square(image.convert("RGB"), fill_color=self.background_color)
        return padded.resize((self.target_size, self.target_size), Image.Resampling.LANCZOS)

    def pad_to_square(self, image: Image.Image, fill_color=(0, 0, 0)) -> Image.Image:
        """Center the image on a `max(width, height)` square canvas filled
        with `fill_color`."""
        width, height = image.size
        max_side = max(width, height)

        padded = Image.new("RGB", (max_side, max_side), fill_color)
        offset_x = (max_side - width) // 2
        offset_y = (max_side - height) // 2
        padded.paste(image, (offset_x, offset_y))
        return padded
