from collections import namedtuple
from collections.abc import Callable

from typing import Dict

from pathlib import Path

from datasets import load_dataset

from transformers import AutoImageProcessor
from transformers import (
    AutoConfig,
    AutoModelForImageClassification,
    ConvNextV2ForImageClassification,
    CLIPForImageClassification,
    LevitForImageClassification,
    MobileViTV2Config,
    ResNetForImageClassification,
    SegformerForSemanticSegmentation,
    Swinv2ForImageClassification,
    ViTForImageClassification,
)

import pytorch2timeloop

PyTorchModel = namedtuple("PyTorchModel", "model inputs")


def clip(image) -> PyTorchModel:
    model_id = "openai/clip-vit-base-patch32"
    model = CLIPForImageClassification(AutoConfig.from_pretrained(model_id))

    image_processor = AutoImageProcessor.from_pretrained(model_id)
    inputs = image_processor(image, return_tensors="pt")
    return PyTorchModel(model, inputs)


def convnextv2(image) -> PyTorchModel:
    model_id = "facebook/convnextv2-tiny-1k-224"
    model = ConvNextV2ForImageClassification(AutoConfig.from_pretrained(model_id))

    image_processor = AutoImageProcessor.from_pretrained(model_id)
    inputs = image_processor(image, return_tensors="pt")
    return PyTorchModel(model, inputs)


def levit(image) -> PyTorchModel:
    model_id = "facebook/levit-128S"
    model = LevitForImageClassification(AutoConfig.from_pretrained(model_id))

    image_processor = AutoImageProcessor.from_pretrained(model_id)
    inputs = image_processor(image, return_tensors="pt")
    return PyTorchModel(model, inputs)


def mobilevit2(image) -> PyTorchModel:
    # Using the timm version for mobilevit2 because the model from Apple requires a token
    model = AutoModelForImageClassification.from_config(MobileViTV2Config())
    image_processor = AutoImageProcessor.from_pretrained(
        "timm/mobilevitv2_100.cvnets_in1k"
    )

    inputs = image_processor(image, return_tensors="pt")
    return PyTorchModel(model, inputs)


def resnet50(image) -> PyTorchModel:
    model_id = "microsoft/resnet-50"
    model = ResNetForImageClassification(AutoConfig.from_pretrained(model_id))

    image_processor = AutoImageProcessor.from_pretrained(model_id)
    inputs = image_processor(image, return_tensors="pt")
    return PyTorchModel(model, inputs)


def segformer(image) -> PyTorchModel:
    model_id = "nvidia/segformer-b0-finetuned-ade-512-512"
    model = SegformerForSemanticSegmentation(AutoConfig.from_pretrained(model_id))

    image_processor = AutoImageProcessor.from_pretrained(model_id)
    inputs = image_processor(images=image, return_tensors="pt")
    return PyTorchModel(model, inputs)


def swinv2(image) -> PyTorchModel:
    model_id = "microsoft/swinv2-tiny-patch4-window8-256"
    model = Swinv2ForImageClassification(AutoConfig.from_pretrained(model_id))

    image_processor = AutoImageProcessor.from_pretrained(model_id)
    inputs = image_processor(images=image, return_tensors="pt")
    return PyTorchModel(model, inputs)


def vitbase(image) -> PyTorchModel:
    model_id = "google/vit-base-patch16-224"
    model = ViTForImageClassification(AutoConfig.from_pretrained(model_id))

    image_processor = AutoImageProcessor.from_pretrained(model_id)
    inputs = image_processor(images=image, return_tensors="pt")
    return PyTorchModel(model, inputs)


def get_model(model: str):
    models_map: Dict[str, Callable[[str], PyTorchModel]] = {
        "clip": clip,
        "convnextv2": convnextv2,
        "levit": levit,
        "mobilevit2": mobilevit2,
        "resnet50": resnet50,
        "segformer": segformer,
        "vitbase": vitbase,
    }

    dataset = load_dataset("huggingface/cats-image")
    image = dataset["test"]["image"][0]

    return models_map[model](image)


if __name__ == "__main__":
    models = [
        "resnet50",
        "clip",
        "convnextv2",
        "levit",
        "mobilevit2",
        "segformer",
        "vitbase",
    ]

    batch_sizes = [1, 8, 16, 64, 128, 256, 512, 1024]

    results_dir = Path("workloads")
    results_dir.mkdir(exist_ok=True)

    for model_name in models:
        print(f"Concerting model {model_name}")
        for batch_size in batch_sizes:
            model_and_input = get_model(model_name)
            model_module = model_and_input.model
            inputs = model_and_input.inputs

            inputs = inputs["pixel_values"].repeat(batch_size, 1, 1, 1)

            try:
                pytorch2timeloop.convert_model_with_sample_input(
                    model_module,
                    inputs,
                    batch_size,
                    f"{model_name}-{batch_size}",
                    results_dir,
                )
            except (NotImplementedError, AttributeError) as e:
                print(f"Received error for {model_name}: {e}")
                break
