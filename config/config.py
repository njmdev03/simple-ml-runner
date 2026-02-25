from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union, get_type_hints, get_origin, get_args
import os
from datetime import datetime
from string import Template
from enum import Enum
from pathlib import Path
from torch import nn
from torch.utils.data import Dataset
from torch.optim import Optimizer
import importlib

from engine.loader import load_from_pyscript
from .factories import get_model_builder


class LazyObject:
    """Proxy that delays loading until first attribute access or call."""
    def __init__(self, loader_callable):
        self._loader = loader_callable
        self._obj = None

    def _ensure(self):
        if self._obj is None:
            self._obj = self._loader()
        return self._obj

    def resolve(self):
        return self._ensure()

    def __getattr__(self, item):
        obj = self._ensure()
        return getattr(obj, item)

    def __call__(self, *args, **kwargs):
        obj = self._ensure()
        return obj(*args, **kwargs)

    def __repr__(self):
        return f"<LazyObject resolved={self._obj is not None}>"


class DefaultValue:
    def __init__(self, value):
        self.value = value

    @staticmethod
    def field(val):
        return field(default_factory=lambda: DefaultValue(val))


class HaltCondition(Enum):
    ACCURACY = "Accuracy"
    LOSS = "Loss"
    TIME = "Time"


class LogLevel(Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"



class Visualizations(Enum):
    ALL = "All"
    LOSS_ACC = "Loss and Accuracy"
    LOSS = "Loss"
    ACC = "ACCURACY"
    DURATIONS = "Epoch Durations"
    TIMING = "Timing"
    SAMPLES = "Samples"
    ARCHITECTURE = "Model Architecture"


class Datasets(Enum):
    TRAINING = "Training"
    TESTING = "Testing"


class TaskType(Enum):
    CLASSIFICATION = "classification"
    SEGMENTATION = "segmentation"
    DETECTION = "detection"


class ResolvedConfig:
    # Meta
    LOG_LEVEL: LogLevel
    LOG_DIR: Optional[Path]
    LOG_NAME: Optional[str]
    PROFILE: bool
    PROFILE_OUTPUT: Optional[Path]

    # Model
    MODEL: nn.Module

    # Device
    DEVICES: List[str]

    # Data
    TRAIN_DATASET: Dataset
    TEST_DATASET: Dataset
    FINAL_OUTPUT_PATH: Path

    # Training Flags
    TRAIN: bool
    TEST: bool

    # Training Hyperparameters
    BATCH_SIZE: int
    LEARNING_RATE: float
    EPOCHS: int
    OPTIMIZER: Optimizer
    TRAIN_CRITERION: nn.Module

    # Checkpointing
    CHECK_RATE: int
    CHECK_MODEL_DIR: Path
    CHECK_MODEL_NAME: str
    SAVE_METADATA: bool
    # True, auto resume from latest. String, resume from that path
    RESUME: Union[bool, Path]

    # Early Halt
    EARLY_HALT_CONDITION: Optional[str]
    EARLY_HALT_THRESHOLD: float

    # Testing & Evaluation
    TESTING_BATCH_SIZE: int
    TESTING_CRITERION: List[nn.Module]
    TEST_ON_TRAINING_DATA: bool
    TEST_WHILE_TRAINING: bool
    TEST_CHECKPOINTS: bool
    SAVE_TESTS: Optional[Path]

    # Visualization
    VIS_TYPE: List[Visualizations]
    VIS_OUTPUT_DIR: Optional[Path]
    VIS_FORMAT: str
    VIS_LAYOUT: str
    SHOW: bool
    NUM_SAMPLES: int
    VIS_DATASETS: List[Datasets]
    VIS_METRICS: List[str]

    # New Evaluation System
    TASK_TYPE: TaskType
    EVAL_METRICS: List[str]
    METRICS: Dict[str, Any]  # Dictionary of instantiated metric objects

    # Custom Step Functions & Data Loading
    COLLATE_FN: Optional[Any]   # Optional callable: list of samples -> batch
    TRAIN_STEP_FN: Optional[Any]  # Optional callable: (model, data, target, device) -> (loss, output)
    EVAL_STEP_FN: Optional[Any]   # Optional callable: (model, data, target, device) -> (outputs, targets)


@dataclass
class Config:
    # Meta
    LOG_LEVEL: LogLevel = DefaultValue.field(LogLevel.INFO)
    LOG_DIR: Optional[Path] = DefaultValue.field(None)
    LOG_NAME: Optional[str] = DefaultValue.field(None)
    PROFILE: bool = DefaultValue.field(False)
    PROFILE_OUTPUT: Optional[Path] = DefaultValue.field(None)

    # Model
    MODEL: Any = DefaultValue.field(None)

    # Device
    DEVICES: List[str] = DefaultValue.field(['cpu'])

    # Data
    TRAIN_DATASET: Any = DefaultValue.field(None)
    TEST_DATASET: Any = DefaultValue.field(None)
    FINAL_OUTPUT_PATH: Path = DefaultValue.field(Path('model_final.pt'))

    # Training Flags
    TRAIN: bool = DefaultValue.field(True)
    TEST: bool = DefaultValue.field(True)

    # Training Hyperparameters
    BATCH_SIZE: int = DefaultValue.field(32)
    LEARNING_RATE: float = DefaultValue.field(0.001)
    EPOCHS: int = DefaultValue.field(1)
    OPTIMIZER: Any = DefaultValue.field('Adam')
    TRAIN_CRITERION: Any = DefaultValue.field('CrossEntropyLoss')

    # Checkpointing
    CHECK_RATE: int = DefaultValue.field(0)
    CHECK_MODEL_DIR: Path = DefaultValue.field(Path('checkpoints/'))
    CHECK_MODEL_NAME: str = DefaultValue.field('model_epoch_$epoch')
    SAVE_METADATA: bool = DefaultValue.field(True)
    # True, auto resume from latest. String, resume from that path
    RESUME: Any = DefaultValue.field(False)

    # Early Halt
    EARLY_HALT_CONDITION: str = DefaultValue.field(None)
    EARLY_HALT_THRESHOLD: float = DefaultValue.field(0.0)

    # Testing & Evaluation
    TESTING_BATCH_SIZE: Optional[int] = DefaultValue.field(None)
    TESTING_CRITERION: Optional[Any] = DefaultValue.field(None)
    TEST_ON_TRAINING_DATA: bool = DefaultValue.field(False)
    TEST_WHILE_TRAINING: bool = DefaultValue.field(False)
    TEST_CHECKPOINTS: bool = DefaultValue.field(False)
    SAVE_TESTS: Optional[Path] = DefaultValue.field(None)

    # New Evaluation System
    TASK_TYPE: str = DefaultValue.field("classification")
    EVAL_METRICS: List[str] = DefaultValue.field(["Accuracy"])
    CUSTOM_METRICS: Dict[str, Any] = DefaultValue.field(None)

    # Custom Step Functions & Data Loading
    COLLATE_FN: Optional[Any] = DefaultValue.field(None)
    TRAIN_STEP_FN: Optional[Any] = DefaultValue.field(None)
    EVAL_STEP_FN: Optional[Any] = DefaultValue.field(None)

    # Visualization
    VIS_TYPE: List[str] = DefaultValue.field(['all'])
    VIS_OUTPUT_DIR: Path = DefaultValue.field(Path('vis'))
    VIS_FORMAT: str = DefaultValue.field('png')
    VIS_LAYOUT: str = DefaultValue.field('individual')
    SHOW: bool = DefaultValue.field(False)
    NUM_SAMPLES: int = DefaultValue.field(0)
    VIS_DATASETS: List[str] = DefaultValue.field(['testing'])
    VIS_METRICS: List[str] = DefaultValue.field(['all'])
    _run_start_time: datetime = field(default_factory=datetime.now, repr=False)

    # Internal: keep track of which fields came from config files
    # _source: Dict[str, str] = field(default_factory=dict, repr=False)

    @classmethod
    def from_dict(cls, d: Dict[str, Any], base_path: Optional[Path] = None) -> 'Config':
        inst = cls()
        inst.update_from_dict(d, base_path=base_path)
        return inst

    def update_from_dict(self, d: Dict[str, Any], base_path: Optional[Path] = None):
        # Normalize keys
        for k, v in d.items():
            if k == 'CONFIG':
                continue

            key = k.upper().replace('-', '_')

            # Resolve file paths relative to base_path for keys annotated as Path
            if isinstance(v, (str, Path)) and base_path:
                if key in _PATH_LIKE_FIELDS:
                    p = Path(v)
                    if not p.is_absolute():
                        v = Path((base_path / p).resolve())

            # Special handling for MODEL
            if key == 'MODEL':
                self.MODEL = self._parse_model_spec(v)
                # self._source['MODEL'] = str(base_path) if base_path else '<cli>'
                continue

            if key == 'TRAIN_CRITERION':
                self.TRAIN_CRITERION = self._parse_criterion_spec(v)
                # self._source['TRAIN_CRITERION'] = str(base_path) if base_path else '<cli>'
                continue

            if key == 'TESTING_CRITERION':
                crits = []

                for crit in v:
                    crits.append(self._parse_criterion_spec(crit))

                self.TESTING_CRITERION = crits

            if key in ('TRAIN_DATASET', 'TEST_DATASET'):
                self.__setattr__(key, self._parse_dataset_spec(v))
                # self._source[key] = str(base_path) if base_path else '<cli>'
                continue

            # Generic assignment if attribute exists
            if hasattr(self, key):
                try:
                    # Basic casting for primitives
                    if isinstance(getattr(self, key), bool):
                        if isinstance(v, str):
                            lv = v.strip().lower()
                            if lv in ('true', '1', 'yes', 'on'):
                                v = True
                            elif lv in ('false', '0', 'no', 'off'):
                                v = False

                    setattr(self, key, v)
                    # self._source[key] = str(base_path) if base_path else '<cli>'
                except Exception:
                    setattr(self, key, v)
            else:
                # Unknown key: set as attribute for flexibility
                setattr(self, key, v)

    # Merge the overlay config with this config. Values defined in the overlay
    # Replace values in this config
    def merge(self, overlay: 'Config'):
        """Merge values from `overlay` into this `Config`.

        Overlay values that are stored as `DefaultValue` instances are
        treated as defaults and do NOT overwrite existing values.
        """
        ov_dict = getattr(overlay, '__dict__', {})
        for k, raw_val in ov_dict.items():
            # skip private/internal attrs
            if k.startswith('_'):
                continue

            # If overlay stored a DefaultValue wrapper, skip it
            if isinstance(raw_val, DefaultValue):
                continue

            # Otherwise copy the raw value to self (best-effort)
            try:
                setattr(self, k, raw_val)
            except Exception:
                object.__setattr__(self, k, raw_val)

            # Propagate provenance if present on overlay
            # try:
            #     if hasattr(self, '_source') and hasattr(overlay, '_source'):
            #         src = getattr(overlay, '_source')
            #         if isinstance(src, dict) and k in src:
            #             self._source[k] = src[k]
            # except Exception:
            #     pass

    def resolve_all(self):
        # Resolve any LazyObjects to their underlying objects
        for k, v in list(self.__dict__.items()):
            if isinstance(v, LazyObject):
                try:
                    resolved = v.resolve()
                    setattr(self, k, resolved)
                except Exception:
                    # keep as LazyObject if resolution fails
                    pass

    # Merge the overlay config into this config. Values defined in the overlay
    # replace values in this config, except when the overlay value is a
    # `DefaultValue` instance (those indicate a default and should not
    # overwrite existing explicit values).
    def merge(self, overlay: 'Config') -> 'Config':
        # Iterate keys present on the overlay instance (use __dict__ to
        # observe raw stored values and avoid Config.__getattribute__ resolution).
        ov_dict = getattr(overlay, '__dict__', {})
        for k, raw_val in ov_dict.items():
            # skip private/internal attrs
            if k.startswith('_'):
                continue

            # If the overlay stored a DefaultValue wrapper, skip it
            if isinstance(raw_val, DefaultValue):
                continue

            # Otherwise, copy the raw value to self
            try:
                setattr(self, k, raw_val)
            except Exception:
                # best-effort: set attribute even if setter misbehaves
                object.__setattr__(self, k, raw_val)

            # Propagate provenance information if present
            # try:
            #     if hasattr(self, '_source') and hasattr(overlay, '_source'):
            #         src = getattr(overlay, '_source')
            #         if isinstance(src, dict) and k in src:
            #             self._source[k] = src[k]
            # except Exception:
            #     pass

        return self

    # Resolve the current config to a Resolved config with stricter types.
    def resolve(self) -> ResolvedConfig:
        # Ensure conditional defaults are applied first
        # try:
        #     self.finalize()
        # except Exception:
        #     pass
        # Do not call `finalize()` here; resolved defaults for testing
        # should be applied onto the ResolvedConfig so callers always get
        # concrete testing values without mutating the original Config.

        rc = ResolvedConfig()

        for name, tp in get_type_hints(ResolvedConfig).items():
            # Read the user-facing value (this goes through __getattribute__ and
            # will unwrap DefaultValue wrappers)
            val = getattr(self, name, None)

            # MODEL: allow LazyObject or parsed model spec
            if name == 'MODEL':
                if isinstance(val, LazyObject):
                    setattr(rc, name, val)
                else:
                    setattr(rc, name, self._parse_model_spec(val))
                continue

            # Datasets: keep LazyObject or parse dataset spec
            if name in ('TRAIN_DATASET', 'TEST_DATASET'):
                if isinstance(val, LazyObject):
                    setattr(rc, name, val)
                else:
                    setattr(rc, name, self._parse_dataset_spec(val))
                continue

            # Criteria: parse strings/dicts into criterion objects; allow None
            if name in ('TRAIN_CRITERION', 'TESTING_CRITERION'):
                if val is None:
                    setattr(rc, name, None)
                elif isinstance(val, (list, tuple)):
                    setattr(rc, name, [self._parse_criterion_spec(v) for v in val])
                else:
                    setattr(rc, name, self._parse_criterion_spec(val))
                continue

            # Path-like annotated fields -> ensure Path instances (or None)
            if _is_path_type(tp):
                if val is None:
                    setattr(rc, name, None)
                else:
                    # Apply template resolution to paths
                    if isinstance(val, (str, Path)):
                        val = self._resolve_templates(str(val))

                    if isinstance(val, Path):
                        setattr(rc, name, val)
                    elif isinstance(val, (str, bytes, os.PathLike)):
                        setattr(rc, name, Path(val))
                    else:
                        setattr(rc, name, val)
                continue

            # Enum Handling
            origin = get_origin(tp)
            if origin in (list, List):
                args = get_args(tp)
                if args and isinstance(args[0], type) and issubclass(args[0], Enum):
                    enum_cls = args[0]
                    if isinstance(val, (list, tuple)):
                        resolved_list = []
                        for item in val:
                            if isinstance(item, str):
                                try:
                                    resolved_list.append(enum_cls(item) if any(e.value == item for e in enum_cls) else enum_cls[item.upper()])
                                except:
                                    # Fallback: try case-insensitive or ignore
                                    resolved_list.append(item)
                            else:
                                resolved_list.append(item)
                        setattr(rc, name, resolved_list)
                    else:
                        setattr(rc, name, val)
                    continue
            elif isinstance(tp, type) and issubclass(tp, Enum):
                if isinstance(val, str):
                    try:
                        setattr(rc, name, tp(val) if any(e.value == val for e in tp) else tp[val.upper()])
                    except:
                        setattr(rc, name, val)
                else:
                    setattr(rc, name, val)
                continue

            # Default: copy the value through (LazyObjects preserved)
            # Also apply template resolution to strings that aren't Paths but might be templates
            if isinstance(val, str) and name != 'CHECK_MODEL_NAME':
                val = self._resolve_templates(val)

            setattr(rc, name, val)

        # Apply testing fallbacks on the resolved config (do not change
        # the original `Config` instance).
        if getattr(rc, 'TESTING_BATCH_SIZE', None) is None:
            rc.TESTING_BATCH_SIZE = rc.BATCH_SIZE

        if getattr(rc, 'TESTING_CRITERION', None) is None:
            rc.TESTING_CRITERION = [rc.TRAIN_CRITERION]

        # Finalize Metrics Setup
        rc.TASK_TYPE = TaskType(self.TASK_TYPE.lower())
        rc.EVAL_METRICS = self.EVAL_METRICS

        # Instantiate Metrics
        rc.METRICS = {}
        if self.CUSTOM_METRICS:
            rc.METRICS.update(self.CUSTOM_METRICS)
        else:
            from utils.metrics import Accuracy, Precision, Recall, F1Score, MeanIoU, PixelAccuracy, DetectionMAP

            metric_map = {
                "Accuracy": Accuracy,
                "Precision": Precision,
                "Recall": Recall,
                "F1": F1Score,
                "MeanIoU": MeanIoU,
                "PixelAccuracy": PixelAccuracy,
                "DetectionMAP": DetectionMAP,
            }

            for m_name in rc.EVAL_METRICS:
                if m_name in metric_map:
                    rc.METRICS[m_name] = metric_map[m_name]()
                else:
                    logger.warning(f"Unknown metric '{m_name}' for task type '{rc.TASK_TYPE.value}'")

        return rc

    def _resolve_templates(self, val: str) -> str:
        if '$' not in val:
            return val
        now = self._run_start_time
        return Template(val).safe_substitute(
            date=now.strftime("%Y-%m-%d"),
            time=now.strftime("%H-%M-%S")
        )

    def get(self, key: str, default: Any = None):
        key_u = key.upper()
        if hasattr(self, key_u):
            return getattr(self, key_u)
        return default

    def to_dict(self) -> Dict[str, Any]:
        # Use asdict but avoid trying to serialize LazyObjects
        out = {}
        for f, v in self.__dict__.items():
            if f.startswith('_'):
                continue
            if isinstance(v, LazyObject):
                out[f] = repr(v)
            else:
                out[f] = v
        return out

    def _parse_model_spec(self, spec: Any):
        # If it's already an object, return
        if spec is None:
            return None
        # If it's a LazyObject already
        if isinstance(spec, LazyObject):
            return spec

        # String or Path handling
        if isinstance(spec, (str, Path)):
            # If points to a .py file or filesystem path -> lazy load from script
            p = Path(spec)
            if p.suffix == '.py' or p.exists():
                return LazyObject(lambda: load_from_pyscript(str(p), ["MODEL", "Net"]))

            # Try dotted import: attempt to import attribute
            if '.' in spec:
                def _imp():
                    module_name, _, attr = spec.rpartition('.')
                    mod = importlib.import_module(module_name)
                    return getattr(mod, attr)
                return LazyObject(_imp)

            # Fallback: plain string -> treat as class name in torch.nn
            return LazyObject(lambda: getattr(importlib.import_module('torch.nn'), spec))

        # If it's a dict describing a model: support {type:..., params:...} or {MLP: {...}}
        if isinstance(spec, dict):
            # inline shorthand
            if 'type' in spec:
                typ = spec['type']
                params = spec.get('params', {})
            elif 'class' in spec:
                typ = spec['class']
                params = spec.get('params', {})
            else:
                # shorthand {"MLP": {...}}
                keys = list(spec.keys())
                if len(keys) == 1:
                    typ = keys[0]
                    params = spec[keys[0]] or {}
                else:
                    raise ValueError(f"Invalid model dict spec: {spec}")

            # If typ looks like dotted import
            if isinstance(typ, str) and '.' in typ:
                def _imp_build():
                    module_name, _, attr = typ.rpartition('.')
                    mod = importlib.import_module(module_name)
                    cls = getattr(mod, attr)
                    return cls(**params) if callable(cls) else cls
                return LazyObject(_imp_build)

            # Try registered builder
            builder = get_model_builder(str(typ))
            if builder:
                return builder(params)

            raise ValueError(f"Unknown model type '{typ}'")

        # If it's a type/class already
        return spec

    def _parse_criterion_spec(self, spec: Any):
        if spec is None:
            return None
        if isinstance(spec, LazyObject):
            return spec

        if isinstance(spec, (str, Path)):
            # dotted import
            if isinstance(spec, str) and '.' in spec:
                def _imp():
                    module_name, _, attr = spec.rpartition('.')
                    mod = importlib.import_module(module_name)
                    cls = getattr(mod, attr)
                    return cls()
                return LazyObject(_imp)

            # try torch.nn
            try:
                import torch.nn as nn
                if hasattr(nn, spec):
                    return getattr(nn, spec)()
            except Exception:
                pass

            # fallback: return as-is
            return spec

        if isinstance(spec, dict):
            # support {class: 'm.module.Class', params: {...}} or {type: 'BCELoss'} etc.
            if 'class' in spec:
                clsname = spec['class']
                params = spec.get('params', {})
                module_name, _, attr = clsname.rpartition('.')
                mod = importlib.import_module(module_name)
                cls = getattr(mod, attr)
                return cls(**params)
        return spec

    def _parse_dataset_spec(self, spec: Any):
        # If string/path and looks like .py file or path -> lazy load
        if isinstance(spec, (str, Path)):
            p = Path(spec)
            if p.suffix == '.py' or p.exists():
                return LazyObject(lambda: load_from_pyscript(str(p), ["TRAIN_DATASET", "TEST_DATASET", "Dataset", "dataset"]))
            # dotted imports can be supported in future
            return spec

        # if dict with some dataset params, keep as-is for future factories
        return spec

    def __getattribute__(self, name):
        value = object.__getattribute__(self, name)
        if isinstance(value, DefaultValue):
            return value.value
        return value


# Build a cached set of config keys whose annotated type refers to pathlib.Path
def _is_path_type(tp) -> bool:
    """Return True if the annotation `tp` represents Path or Optional[Path] / Union[..., Path]."""
    origin = get_origin(tp)
    if origin is Union:
        return any(_is_path_type(a) for a in get_args(tp))

    # direct Path
    try:
        if tp is Path or tp is Any:
            return True
    except Exception:
        pass

    return False


# Compute path-like fields based on the `Config` class annotations so we treat
# string incoming values for those keys as filesystem paths relative to
# `base_path` when present. Include a few extra keys that should be
# interpreted as paths even if not annotated as `Path` (e.g. MODEL and
# dataset script/file references).
# _EXTRA_PATH_KEYS = {"MODEL", "TRAIN_DATASET", "TEST_DATASET"}
_PATH_LIKE_FIELDS = {k.upper() for k, v in get_type_hints(Config).items() if _is_path_type(v)}
