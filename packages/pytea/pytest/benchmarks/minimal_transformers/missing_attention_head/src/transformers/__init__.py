# flake8: noqa
# There's no way to ignore "F401 '...' imported but unused" warnings in this
# module, but to preserve other warnings. So, don't check this module at all.

__version__ = "3.5.0"

# Configurations
from .configuration_bert import BERT_PRETRAINED_CONFIG_ARCHIVE_MAP, BertConfig
from .configuration_utils import PretrainedConfig

from .modeling_bert import (
    BertForSequenceClassification, BertModel,
)
from .modeling_utils import PreTrainedModel

# Optimization
from .optimization import (
    AdamW, get_linear_schedule_with_warmup,
)

