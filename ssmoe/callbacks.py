from typing import Dict, Any

from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments


class MoELoggingCallback(TrainerCallback):
    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        model = kwargs.get('model', None)
        if model is None:
            return
        logs: Dict[str, Any] = {}
        util = getattr(model, '_last_moe_utilization', None)
        if util is not None:
            try:
                util = util.detach().cpu().tolist()
                for i, v in enumerate(util):
                    logs[f"moe/util_expert_{i}"] = float(v)
            except Exception:
                pass
        bal = getattr(model, '_last_moe_balance', None)
        if bal is not None:
            try:
                logs["moe/balance_loss"] = float(bal)
            except Exception:
                pass
        if logs:
            kwargs.get('trainer', None).log(logs)

