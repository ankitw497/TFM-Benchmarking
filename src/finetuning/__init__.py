"""
Fine-tuning module for tabular foundation models.

Provides infrastructure for:
- TabPFN v2/v2.5 gradient-based fine-tuning (TabPFNFineTuner)
- TabNet self-supervised pretraining + supervised fine-tuning (TabNetFineTuner)
- FT-Transformer full training loop with LR scheduler (FTTransformerFineTuner)
"""

from .trainer import FineTuningConfig, FineTuningResult, BaseFineTuner
from .tabpfn_finetune import TabPFNFineTuner
from .tabnet_finetune import TabNetFineTuner
from .ft_transformer_finetune import FTTransformerFineTuner

__all__ = [
    "FineTuningConfig",
    "FineTuningResult",
    "BaseFineTuner",
    "TabPFNFineTuner",
    "TabNetFineTuner",
    "FTTransformerFineTuner",
]
