import inspect

import torch.optim

d = {}

for name, class_type in inspect.getmembers(torch.optim):
    if inspect.isclass(class_type) and issubclass(class_type, torch.optim.Optimizer) and class_type is not torch.optim.Optimizer:
        d[class_type] = {}
        cls_init_sig: inspect.Signature = inspect.signature(class_type.__init__)
        for param_name, param in cls_init_sig.parameters.items():
            if param_name not in ("self", "params"):
                d[class_type][param_name] = {
                    "default": param.default,
                    "type": param.annotation if param.annotation is not param.empty else type(param.default),
                }

print(d)