from pathlib import Path

from PIL import Image
from tqdm import tqdm

from pipeline.preprocessing.preprocessing_step import PreprocessingStep


class Normalizer(PreprocessingStep):
    """Pad images to a square canvas, resize them to 256x256, and save them.

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
        **kwargs,
    ):
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

    def run(self) -> Path:
        """Pad each image to a square, resize to 256x256, and save under output_path."""
        output_dir = Path(self.output_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        image_paths = self._collect_image_paths(self.input_path)
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
        """Pad the image with a background color to the nearest square size."""
        width, height = image.size
        max_side = max(width, height)

        padded = Image.new("RGB", (max_side, max_side), fill_color)
        offset_x = (max_side - width) // 2
        offset_y = (max_side - height) // 2
        padded.paste(image, (offset_x, offset_y))
        return padded

    def _collect_image_paths(self, input_value) -> list[Path]:
        """Collect every existing image path from a file/dir or a list/tuple of them."""
        candidates = input_value if isinstance(input_value, (list, tuple)) else [input_value]
        collected: list[Path] = []
        seen: set[str] = set()

        for raw_path in candidates:
            if raw_path is None:
                continue

            path = Path(raw_path)
            if not path.exists():
                continue

            if path.is_file():
                if path.suffix.lower() in self.image_extensions:
                    resolved = path.resolve()
                    if str(resolved) not in seen:
                        seen.add(str(resolved))
                        collected.append(path)
                continue

            for nested in sorted(path.rglob("*")):
                if not nested.is_file():
                    continue
                if nested.suffix.lower() not in self.image_extensions:
                    continue
                resolved = nested.resolve()
                if str(resolved) not in seen:
                    seen.add(str(resolved))
                    collected.append(nested)

        return collected
