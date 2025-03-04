import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .quantization import quantize_weight, dequantize_weight

class QLISADiffusion:
    def __init__(self, model, rate=None):
        """
        Initialize QLISA by quantizing the model's weights and setting up selective layer updating.
        :param model: A PyTorch model (e.g. a diffusion model) to apply QLISA on.
        :param rate: Optional override for the fraction of intermediate layers to activate.
        """
        self.model = model
        self.rate = rate
        # Pre-quantize all linear layers in the model.
        self.apply_quantization(self.model)
        # Prepare dictionaries for optimizer and scheduler hooks.
        self.optimizer_dict = {}
        self.scheduler_dict = {}
        # Apply the initial selective layer update.
        self.qlisa_recall()

    def apply_quantization(self, model):
        """
        For every nn.Linear layer in the model, perform:
          1. Save the full-precision weight.
          2. Compute a scaling factor and quantize the weight to a 4-bit NF4 representation.
          3. Double-quantize the scaling factor.
          4. Replace the forward method with on-the-fly dequantization.
        """
        for module in model.modules():
            if isinstance(module, nn.Linear):
                module.weight_fp32 = module.weight.detach().clone()
                weight_q, scale_fp32, scale_q = quantize_weight(module.weight_fp32)
                module.weight_q = weight_q
                module.scale_fp32 = scale_fp32
                module.scale_q = scale_q
                module.weight.requires_grad = False

                # Replace forward method to dequantize weights on the fly.
                def quantized_forward(input, module=module):
                    dequant_weight = dequantize_weight(module.weight_q, module.scale_fp32, module.scale_q)
                    return F.linear(input, dequant_weight, module.bias)
                module.forward = quantized_forward

    def freeze_all_layers(self, model):
        """Freeze all parameters in the model."""
        for param in model.parameters():
            param.requires_grad = False

    def random_activate_layers(self, model, p):
        """
        Activate a subset of layers.
        The first and last parameter groups (e.g. embedding and LM head) are always active.
        :param model: The model whose parameters will be updated.
        :param p: Fraction of intermediate layers to activate.
        """
        params = list(model.parameters())
        num_params = len(params)
        activate_number = int((num_params - 2) * p)
        indices = np.random.choice(range(1, num_params - 1), activate_number, replace=False)
        for i, param in enumerate(params):
            if i == 0 or i == num_params - 1 or i in indices:
                param.requires_grad = True
            else:
                param.requires_grad = False

    def qlisa(self, model, p=0.25):
        """Perform QLISA: freeze all layers, then randomly activate a fraction (p) of intermediate layers."""
        self.freeze_all_layers(model)
        self.random_activate_layers(model, p)

    def qlisa_recall(self):
        """
        Recalculate the selective activation probability and apply QLISA.
        Default is 8/total_params unless overridden.
        """
        num_params = len(list(self.model.parameters()))
        qlisa_p = 8 / num_params if self.rate is None else self.rate
        self.qlisa(self.model, p=qlisa_p)

    def register(self, optimizer_class, get_scheduler, accelerator, optim_kwargs={}, sched_kwargs={}):
        """
        Create and store an optimizer and scheduler for every active parameter.
        :param optimizer_class: The optimizer class (e.g. torch.optim.AdamW).
        :param get_scheduler: Function that returns a scheduler instance.
        :param accelerator: An accelerator (e.g. from Hugging Face Accelerate) to prepare optimizers/schedulers.
        :param optim_kwargs: Additional optimizer kwargs.
        :param sched_kwargs: Additional scheduler kwargs.
        """
        for p in self.model.parameters():
            if p.requires_grad:
                self.optimizer_dict[p] = optimizer_class([{"params": p}], **optim_kwargs)
                if accelerator is not None:
                    self.optimizer_dict[p] = accelerator.prepare_optimizer(self.optimizer_dict[p])
        for p in self.model.parameters():
            if p.requires_grad:
                self.scheduler_dict[p] = get_scheduler(optimizer=self.optimizer_dict[p], **sched_kwargs)
                if accelerator is not None:
                    self.scheduler_dict[p] = accelerator.prepare_scheduler(self.scheduler_dict[p])

    def insert_hook(self, optimizer_class, get_scheduler, accelerator, optim_kwargs={}, sched_kwargs={}):
        """
        Insert a hook that, after gradient accumulation on an active parameter, performs:
          gradient clipping, optimizer step, zeroing, and scheduler stepping.
        """
        def optimizer_hook(p):
            if p.grad is None:
                self.scheduler_dict.pop(p, None)
                self.optimizer_dict.pop(p, None)
                return
            if p not in self.optimizer_dict:
                self.optimizer_dict[p] = optimizer_class([{"params": p}], **optim_kwargs)
                if accelerator is not None:
                    self.optimizer_dict[p] = accelerator.prepare_optimizer(self.optimizer_dict[p])
            if p not in self.scheduler_dict:
                self.scheduler_dict[p] = get_scheduler(optimizer=self.optimizer_dict[p], **sched_kwargs)
                if accelerator is not None:
                    self.scheduler_dict[p] = accelerator.prepare_scheduler(self.scheduler_dict[p])
            if accelerator is not None and accelerator.sync_gradients:
                torch.nn.utils.clip_grad_norm_(p, 10.0)
            self.optimizer_dict[p].step()
            self.optimizer_dict[p].zero_grad(set_to_none=True)
            self.scheduler_dict[p].step()

        for p in self.model.parameters():
            if p.requires_grad:
                p.register_post_accumulate_grad_hook(optimizer_hook)
