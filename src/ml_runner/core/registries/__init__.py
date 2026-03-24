from ml_runner.core.registries.model import ModelRegistry, Model
from ml_runner.core.registries.dataset import DatasetRegistry, Dataset
from ml_runner.core.registries.optimizer import OptimizerRegistry, Optimizer
from ml_runner.core.registries.loss import LossRegistry, Loss
from ml_runner.core.registries.metric import MetricRegistry, Metric
from ml_runner.core.registries.metadata_writer import MetadataWriterRegistry, MetadataWriter
from ml_runner.core.registries.scheduler import SchedulerRegistry, Scheduler
from ml_runner.core.registries.config import ConfigRegistry, Config, NamedConfig, NamedConfigRegistry
from ml_runner.core.registries.config_parser import ConfigParserRegistry, ConfigParser
from ml_runner.core.registries.extension import ExtensionRegistry, Extension
from ml_runner.core.registries.task import TaskRegistry, Task

from ml_runner.core.registries.utils import resolve_component
