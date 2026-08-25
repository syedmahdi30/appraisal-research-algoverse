import importlib
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from src.experiments.shared import hf_runtime
from src.experiments.shared.hf_runtime import (
    capture_residuals,
    encode_image_prompt,
    find_language_layers,
    last_token_tap,
    load_gemma_hf,
    load_vlm,
    patch_residuals,
    resize_long_side,
)


@pytest.fixture(autouse=True)
def _reset_language_layer_cache():
    hf_runtime._LANGUAGE_LAYER_CACHE.clear()
    yield
    hf_runtime._LANGUAGE_LAYER_CACHE.clear()


class Block(torch.nn.Module):
    def __init__(self, tuple_output=False):
        super().__init__()
        self.tuple_output = tuple_output
        self.metadata = object()

    def forward(self, hidden):
        output = hidden + 1
        return (output, self.metadata) if self.tuple_output else output


class TupleTap(torch.nn.Module):
    def forward(self, hidden):
        return hidden + 3, "attention-weights"


class TappedBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.post_attention_layernorm = TupleTap()

    def forward(self, hidden):
        output, _weights = self.post_attention_layernorm(hidden)
        return output


class ToyModel(torch.nn.Module):
    def __init__(self, layers=None):
        super().__init__()
        self.vision_tower = torch.nn.Module()
        self.vision_tower.layers = torch.nn.ModuleList([Block()])
        self.language_model = torch.nn.Module()
        self.language_model.layers = torch.nn.ModuleList(layers or [Block(), Block()])

    def forward(self, hidden):
        for layer in self.language_model.layers:
            output = layer(hidden)
            hidden = output[0] if isinstance(output, tuple) else output
        return hidden


class SingleLayerModel(ToyModel):
    def forward(self, hidden):
        return self.language_model.layers[0](hidden)


class ConversionTrackingTensor(torch.Tensor):
    conversion_calls = []

    @staticmethod
    def __new__(cls, data):
        return torch.as_tensor(data, dtype=torch.float64).as_subclass(cls)

    def to(self, *args, **kwargs):
        type(self).conversion_calls.append((args, kwargs))
        return super().to(*args, **kwargs)


def test_language_layer_discovery_excludes_vision_and_caches_by_model_identity():
    model = ToyModel()
    found = find_language_layers(model, verbose=False)

    assert found is model.language_model.layers
    model.language_model.layers = torch.nn.ModuleList([Block()])
    assert find_language_layers(model, verbose=False) is found


def test_language_layer_discovery_raises_when_no_decoder_layers_exist():
    model = torch.nn.Linear(2, 2)

    with pytest.raises(RuntimeError, match="could not locate the language-model decoder layers"):
        find_language_layers(model, verbose=False)


def test_capture_residuals_caches_full_sequences_and_preserves_tuple_outputs():
    tuple_block = Block(tuple_output=True)
    model = ToyModel([tuple_block, Block()])
    hidden = torch.zeros(1, 3, 2)

    with capture_residuals(model, [0, 1]) as captured:
        output = model(hidden)

    assert captured[0].shape == (3, 2)
    assert captured[0].tolist() == [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]]
    assert captured[1].tolist() == [[2.0, 2.0], [2.0, 2.0], [2.0, 2.0]]
    assert output.tolist() == [[[2.0, 2.0], [2.0, 2.0], [2.0, 2.0]]]


def test_residual_patch_changes_only_batch_zero_mapped_positions_without_mutation():
    model = ToyModel()
    hidden = torch.tensor([
        [[0.0, 0.0], [10.0, 10.0], [20.0, 20.0]],
        [[100.0, 100.0], [110.0, 110.0], [120.0, 120.0]],
    ])
    donor = {0: torch.tensor([[5.0, 5.0], [6.0, 6.0], [7.0, 7.0]])}
    unpatched = []
    observer = model.language_model.layers[0].register_forward_hook(
        lambda _module, _inputs, output: unpatched.append(output)
    )
    try:
        with patch_residuals(model, donor, [1], [2]):
            output = model(hidden)
    finally:
        observer.remove()

    assert unpatched[0][0, 2].tolist() == [21.0, 21.0]
    assert output[0, 0].tolist() == [2.0, 2.0]
    assert output[0, 1].tolist() == [12.0, 12.0]
    assert output[0, 2].tolist() == [7.0, 7.0]
    assert output[1].tolist() == [
        [102.0, 102.0],
        [112.0, 112.0],
        [122.0, 122.0],
    ]


def test_residual_patch_converts_donor_to_recipient_device_and_dtype():
    model = ToyModel([Block()])
    hidden = torch.zeros(1, 3, 2, dtype=torch.float32)
    donor = ConversionTrackingTensor([
        [1.25, 2.5],
        [3.75, 4.5],
        [5.25, 6.5],
    ])
    ConversionTrackingTensor.conversion_calls = []

    with patch_residuals(model, {0: donor}, [1], [2]):
        output = model(hidden)

    assert donor.dtype == torch.float64
    assert output.dtype == torch.float32
    assert output[0, 2].tolist() == [3.75, 4.5]
    assert ConversionTrackingTensor.conversion_calls == [
        ((), {"device": output.device, "dtype": output.dtype})
    ]


def test_residual_patch_preserves_decoder_tuple_tail():
    block = Block(tuple_output=True)
    model = SingleLayerModel([block])

    with patch_residuals(
        model,
        {0: torch.tensor([[5.0, 5.0], [6.0, 6.0], [7.0, 7.0]])},
        [0],
        [1],
    ):
        output = model(torch.zeros(1, 3, 2))

    assert isinstance(output, tuple)
    assert output[1] is block.metadata
    assert output[0][0, 1].tolist() == [5.0, 5.0]


def test_last_token_tap_uses_nested_target_and_tuple_tensor():
    model = ToyModel([TappedBlock()])
    hidden = torch.tensor([[[1.0, 2.0], [10.0, 20.0], [100.0, 200.0]]])

    with last_token_tap(model, 0, "post_attention_layernorm") as store:
        model(hidden)

    assert store[0].dtype == np.float32
    assert store[0].tolist() == [103.0, 203.0]


@pytest.mark.parametrize("helper", ["tap", "capture", "patch"])
def test_hook_contexts_remove_every_hook_when_the_body_raises(helper):
    model = ToyModel([TappedBlock(), Block()])
    target = (
        model.language_model.layers[0].post_attention_layernorm
        if helper == "tap"
        else model.language_model.layers[0]
    )
    context = {
        "tap": lambda: last_token_tap(model, 0, "post_attention_layernorm"),
        "capture": lambda: capture_residuals(model, [0, 1]),
        "patch": lambda: patch_residuals(
            model, {0: torch.zeros(3, 2)}, [0], [1]
        ),
    }[helper]()

    with pytest.raises(RuntimeError, match="stop"):
        with context:
            raise RuntimeError("stop")

    assert not target._forward_hooks
    if helper == "capture":
        assert not model.language_model.layers[1]._forward_hooks


class LoadedModel:
    def __init__(self, loader_name):
        self.loader_name = loader_name
        self.eval_called = False

    def eval(self):
        self.eval_called = True
        return self


def _loader(name, calls):
    class Loader:
        @classmethod
        def from_pretrained(cls, model_name, **kwargs):
            calls.append((name, model_name, kwargs))
            return LoadedModel(name)

    return Loader


def test_load_gemma_hf_preserves_auto_classes_dtype_and_device_map(monkeypatch):
    calls = []
    fake = types.ModuleType("transformers")
    fake.AutoModelForImageTextToText = _loader("gemma-model", calls)
    fake.AutoProcessor = _loader("processor", calls)
    monkeypatch.setitem(sys.modules, "transformers", fake)

    model, processor = load_gemma_hf("google/gemma-3-4b-it")

    assert calls == [
        ("gemma-model", "google/gemma-3-4b-it", {
            "dtype": torch.bfloat16,
            "device_map": "auto",
        }),
        ("processor", "google/gemma-3-4b-it", {}),
    ]
    assert model.eval_called is True
    assert processor.loader_name == "processor"


@pytest.mark.parametrize(
    ("model_name", "expected_loader", "family", "processor_kwargs"),
    [
        ("Qwen/Qwen3-VL-8B-Instruct", "qwen3", "qwen", {"max_pixels": 1024}),
        ("Qwen/Qwen2.5-VL-7B-Instruct", "qwen2.5", "qwen", {"max_pixels": 1024}),
        ("llava-hf/llava-1.5-7b-hf", "auto-image-text", "llava", {}),
    ],
)
def test_load_vlm_preserves_family_dispatch_and_loader_arguments(
    monkeypatch, model_name, expected_loader, family, processor_kwargs
):
    calls = []
    fake = types.ModuleType("transformers")
    fake.Qwen3VLForConditionalGeneration = _loader("qwen3", calls)
    fake.Qwen2_5_VLForConditionalGeneration = _loader("qwen2.5", calls)
    fake.AutoModelForImageTextToText = _loader("auto-image-text", calls)
    fake.LlavaForConditionalGeneration = _loader("llava-fallback", calls)
    fake.AutoProcessor = _loader("processor", calls)
    monkeypatch.setitem(sys.modules, "transformers", fake)

    model, processor, actual_family = load_vlm(model_name, max_side=32)

    assert calls == [
        (expected_loader, model_name, {"torch_dtype": "auto", "device_map": "auto"}),
        ("processor", model_name, processor_kwargs),
    ]
    assert model.eval_called is True
    assert processor.loader_name == "processor"
    assert actual_family == family


def test_load_vlm_uses_llava_loader_fallback_when_auto_class_is_unavailable(
    monkeypatch, capsys
):
    calls = []
    fake = types.ModuleType("transformers")
    fake.LlavaForConditionalGeneration = _loader("llava-fallback", calls)

    class RejectingProcessor:
        @classmethod
        def from_pretrained(cls, model_name, **kwargs):
            calls.append(("processor", model_name, kwargs))
            if kwargs:
                raise TypeError("max_pixels unsupported")
            return LoadedModel("processor")

    fake.AutoProcessor = RejectingProcessor
    monkeypatch.setitem(sys.modules, "transformers", fake)

    model, processor, family = load_vlm("acme/custom-vlm", max_side=16)

    assert calls == [
        ("llava-fallback", "acme/custom-vlm", {
            "torch_dtype": "auto",
            "device_map": "auto",
        }),
        ("processor", "acme/custom-vlm", {}),
    ]
    assert model.loader_name == "llava-fallback"
    assert processor.loader_name == "processor"
    assert family == "llava"
    assert capsys.readouterr().out == ""


def test_load_vlm_warns_and_retries_processor_without_pixel_budget(monkeypatch, capsys):
    calls = []
    fake = types.ModuleType("transformers")
    fake.Qwen3VLForConditionalGeneration = _loader("qwen3", calls)

    class RejectingProcessor:
        @classmethod
        def from_pretrained(cls, model_name, **kwargs):
            calls.append(("processor", model_name, kwargs))
            if kwargs:
                raise ValueError("max_pixels unsupported")
            return LoadedModel("processor")

    fake.AutoProcessor = RejectingProcessor
    monkeypatch.setitem(sys.modules, "transformers", fake)

    load_vlm("Qwen/Qwen3-VL-8B-Instruct", max_side=16)

    assert calls[-2:] == [
        ("processor", "Qwen/Qwen3-VL-8B-Instruct", {"max_pixels": 256}),
        ("processor", "Qwen/Qwen3-VL-8B-Instruct", {}),
    ]
    assert capsys.readouterr().out == (
        "  [warn] processor rejected ['max_pixels']; relying on image resize alone\n"
    )


def test_resize_long_side_preserves_floor_arithmetic_bicubic_and_noop_identity():
    image = Image.new("RGB", (401, 200))
    image.putdata([((x * 17) % 256, (y * 29) % 256, (x + y) % 256)
                   for y in range(200) for x in range(401)])

    resized = resize_long_side(image, 100)
    expected = image.resize((100, 49), Image.BICUBIC)

    assert resized.size == (100, 49)
    assert resized.tobytes() == expected.tobytes()
    assert resize_long_side(image, None) is image
    assert resize_long_side(image, 500) is image


def test_encode_image_prompt_preserves_processor_shape_and_moves_every_tensor():
    calls = []
    image = Image.new("RGB", (2, 2))

    class Processor:
        def __call__(self, **kwargs):
            calls.append(kwargs)
            return {
                "input_ids": torch.tensor([[1, 2]]),
                "pixel_values": torch.tensor([3.0]),
            }

    encoded = encode_image_prompt(Processor(), image, "prompt", torch.device("cpu"))

    assert calls == [{"text": "prompt", "images": [image], "return_tensors": "pt"}]
    assert list(encoded) == ["input_ids", "pixel_values"]
    assert all(value.device.type == "cpu" for value in encoded.values())


def _import_with_blocked_package(module_name, blocked_package):
    script = f"""
import importlib
import importlib.abc
import sys
import types

sys.modules["torch"] = types.ModuleType("torch")

class BlockedPackageFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == {blocked_package!r} or fullname.startswith({blocked_package!r} + "."):
            raise ModuleNotFoundError("blocked import dependency: " + fullname)
        return None

sys.meta_path.insert(0, BlockedPackageFinder())
importlib.import_module({module_name!r})
"""
    return subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).parents[1],
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize("blocked_package", ["transformers", "src.bridge", "sklearn"])
def test_shared_hf_runtime_import_is_lightweight(blocked_package):
    result = _import_with_blocked_package(
        "src.experiments.shared.hf_runtime", blocked_package
    )

    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("module_name", [
    "src.experiments.stage_d_steering_hf",
    "src.experiments.stage_f_patching_hf",
    "src.experiments.stage_f_cross_patching_hf",
    "src.experiments.stage_f_layerwise_hf",
    "src.experiments.stage_f_arbitration_hf",
])
def test_raw_hf_runners_do_not_import_through_stage_c_facade(module_name):
    result = _import_with_blocked_package(
        module_name, "src.experiments.stage_c_transfer_hf"
    )

    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("module_name", [
    "src.experiments.stage_c_transfer",
    "src.experiments.stage_f_patching",
    "src.experiments.stage_f_cross_patching",
    "src.experiments.stage_f_conflict",
])
def test_transformerbridge_runners_do_not_import_shared_hf_runtime(module_name):
    result = _import_with_blocked_package(
        module_name, "src.experiments.shared.hf_runtime"
    )

    assert result.returncode == 0, result.stderr


def test_runner_compatibility_aliases_remain_callable():
    stage_c = importlib.import_module("src.experiments.stage_c_transfer_hf")
    patching = importlib.import_module("src.experiments.stage_f_patching_hf")
    token_budget = importlib.import_module("src.experiments.stage_f_token_budget")

    assert stage_c.find_lm_layers is find_language_layers
    assert stage_c.last_token_tap is last_token_tap
    assert callable(stage_c.load_hf)
    assert patching.resid_capture_full is capture_residuals
    assert patching.encode is encode_image_prompt
    assert token_budget.load_any is load_vlm

    model = ToyModel()
    with patching.patch_resid(
        model,
        donor={0: torch.tensor([[5.0, 5.0], [6.0, 6.0], [7.0, 7.0]])},
        donor_idx=[0],
        recip_idx=[1],
    ):
        output = model(torch.zeros(1, 3, 2))
    assert output[0, 1].tolist() == [6.0, 6.0]

    image = Image.new("RGB", (4, 2))
    assert token_budget._prep_image(img=image, max_side=2).size == (2, 1)
