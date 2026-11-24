from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np

RLDS_ROOT = Path("/home/ubuntu/other_workspace/paper6/openvla-oft-git/datasets/rlds")
SUITES = ("libero_10", "libero_goal", "libero_object", "libero_spatial")

IMAGE_SIZE = 256
STATE_DIM = 8
ACTION_DIM = 7
JOINT_STATE_DIM = 7

_WIRE_VARINT = 0
_WIRE_64BIT = 1
_WIRE_LENGTH = 2
_WIRE_32BIT = 5


def _varint(buffer: bytes, index: int) -> tuple[int, int]:
    result = shift = 0
    while True:
        byte = buffer[index]
        index += 1
        result |= (byte & 0x7F) << shift
        shift += 7
        if not byte & 0x80:
            return result, index


def _wire_fields(buffer: bytes) -> Iterator[tuple[int, Any]]:
    index = 0
    while index < len(buffer):
        key, index = _varint(buffer, index)
        wire_type = key & 7
        field = key >> 3
        if wire_type == _WIRE_LENGTH:
            length, index = _varint(buffer, index)
            yield field, buffer[index : index + length]
            index += length
        elif wire_type == _WIRE_VARINT:
            value, index = _varint(buffer, index)
            yield field, value
        elif wire_type == _WIRE_32BIT:
            yield field, buffer[index : index + 4]
            index += 4
        elif wire_type == _WIRE_64BIT:
            yield field, buffer[index : index + 8]
            index += 8
        else:
            raise ValueError(f"unsupported protobuf wire type {wire_type} for field {field}")


def iter_tfrecords(path: Path) -> Iterator[bytes]:
    with path.open("rb") as handle:
        while True:
            header = handle.read(12)
            if not header:
                return
            if len(header) < 12:
                raise ValueError(f"{path.name}: truncated record header ({len(header)} bytes)")
            length = struct.unpack("<Q", header[:8])[0]
            payload = handle.read(length)
            if len(payload) != length:
                raise ValueError(f"{path.name}: truncated payload, want {length} got {len(payload)}")
            if len(handle.read(4)) != 4:
                raise ValueError(f"{path.name}: missing payload CRC")
            yield payload


def _example_features(record: bytes) -> dict[str, bytes]:
    features: dict[str, bytes] = {}
    for example_field, features_message in _wire_fields(record):
        if example_field != 1:
            continue
        for features_field, entry in _wire_fields(features_message):
            if features_field != 1:
                continue
            parts = dict(_wire_fields(entry))
            key, value = parts.get(1), parts.get(2)
            if isinstance(key, bytes) and isinstance(value, bytes):
                features[key.decode("utf-8")] = value
    return features


def _bytes_list(feature: bytes) -> list[bytes]:
    out: list[bytes] = []
    for field, payload in _wire_fields(feature):
        if field != 1:
            continue
        for inner, item in _wire_fields(payload):
            if inner == 1:
                out.append(item)
    return out


def _float_list(feature: bytes) -> list[float]:
    out: list[float] = []
    for field, payload in _wire_fields(feature):
        if field != 2:
            continue
        for inner, item in _wire_fields(payload):
            if inner != 1:
                continue
            if len(item) % 4:
                raise ValueError(f"packed float blob is not a multiple of 4 bytes: {len(item)}")
            out.extend(struct.unpack(f"<{len(item) // 4}f", item))
    return out


@dataclass(frozen=True)
class Episode:

    suite: str
    shard: str
    index_in_shard: int
    instruction: str
    images: list[bytes]
    wrist_images: list[bytes]
    states: np.ndarray
    joint_states: np.ndarray
    actions: np.ndarray

    def __len__(self) -> int:
        return len(self.images)


def parse_episode(record: bytes, *, suite: str, shard: str, index_in_shard: int) -> Episode:
    features = _example_features(record)
    required = (
        "steps/observation/image",
        "steps/observation/wrist_image",
        "steps/observation/state",
        "steps/observation/joint_state",
        "steps/action",
        "steps/language_instruction",
    )
    missing = [key for key in required if key not in features]
    if missing:
        raise KeyError(f"{shard}[{index_in_shard}]: missing RLDS features {missing}")

    images = _bytes_list(features["steps/observation/image"])
    wrist = _bytes_list(features["steps/observation/wrist_image"])
    instructions = _bytes_list(features["steps/language_instruction"])
    steps = len(images)
    if steps == 0:
        raise ValueError(f"{shard}[{index_in_shard}]: episode has no steps")

    def matrix(key: str, width: int) -> np.ndarray:
        flat = np.asarray(_float_list(features[key]), dtype=np.float32)
        if flat.size != steps * width:
            raise ValueError(
                f"{shard}[{index_in_shard}]: {key} has {flat.size} floats, "
                f"want {steps} steps x {width}"
            )
        return flat.reshape(steps, width)

    if len(wrist) != steps or len(instructions) != steps:
        raise ValueError(
            f"{shard}[{index_in_shard}]: step count disagrees — image {steps}, "
            f"wrist {len(wrist)}, instruction {len(instructions)}"
        )
    actions = matrix("steps/action", ACTION_DIM)
    unique = {value for value in instructions}
    if len(unique) != 1:
        raise ValueError(f"{shard}[{index_in_shard}]: {len(unique)} distinct instructions in one episode")
    return Episode(
        suite=suite,
        shard=shard,
        index_in_shard=index_in_shard,
        instruction=instructions[0].decode("utf-8"),
        images=images,
        wrist_images=wrist,
        states=matrix("steps/observation/state", STATE_DIM),
        joint_states=matrix("steps/observation/joint_state", JOINT_STATE_DIM),
        actions=actions,
    )


def shard_paths(suite: str, root: Path = RLDS_ROOT) -> list[Path]:
    if suite not in SUITES:
        raise ValueError(f"unknown suite {suite!r}; built suites are {SUITES}")
    directory = root / f"{suite}_no_noops" / "1.0.0"
    paths = sorted(directory.glob("*-train.tfrecord-*"))
    if not paths:
        raise FileNotFoundError(f"no tfrecord shards under {directory}")
    prefixes = {path.name.split("-train.tfrecord-")[0] for path in paths}
    if len(prefixes) != 1:
        raise ValueError(f"{directory}: {len(prefixes)} shard prefixes present: {sorted(prefixes)}")
    return paths


def iter_episodes(suite: str, root: Path = RLDS_ROOT) -> Iterator[Episode]:
    for path in shard_paths(suite, root):
        for index, record in enumerate(iter_tfrecords(path)):
            yield parse_episode(record, suite=suite, shard=path.name, index_in_shard=index)


def decode_jpeg(payload: bytes) -> np.ndarray:
    import io

    from PIL import Image

    with Image.open(io.BytesIO(payload)) as handle:
        image = np.asarray(handle.convert("RGB"), dtype=np.uint8)
    if image.shape != (IMAGE_SIZE, IMAGE_SIZE, 3):
        raise ValueError(f"expected {IMAGE_SIZE}x{IMAGE_SIZE}x3 uint8, got {image.shape}")
    return image
