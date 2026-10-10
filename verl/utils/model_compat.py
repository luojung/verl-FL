# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Local Transformers compatibility for causal and vision-language text models.

Do not alias exports on ``transformers``: other libraries must see its real API.
The VLM selector prefers the modern auto class, but retains the legacy class
when it is the only available (or the only registered) implementation.
"""

import transformers

_MEDGEMMA_LANGUAGE_LINEAR = (
    r"model\.language_model\.layers\..*\."
    r"(?:q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"
)
_VLM_AUTO_NAMES = ("AutoModelForImageTextToText", "AutoModelForVision2Seq")


def _vlm_auto_classes():
    return [cls for name in _VLM_AUTO_NAMES if (cls := getattr(transformers, name, None)) is not None]


def _get_vlm_auto_class(config):
    classes = _vlm_auto_classes()
    for cls in classes:
        if type(config) in getattr(cls, "_model_mapping", {}):
            return cls
    if classes:
        return classes[0]
    raise ImportError("This Transformers installation has no supported vision-language AutoModel class")


def get_auto_model_class(config):
    """Select an auto head without requiring either version-specific VLM export.

    Remote auto maps take precedence when they match a declared architecture.
    Missing architecture metadata is supported through native config mappings;
    unknown/base models keep the historical ``AutoModel`` fallback.
    """
    architectures = getattr(config, "architectures", None) or ()
    if isinstance(architectures, str):
        architectures = (architectures,)
    auto_map = getattr(config, "auto_map", None) or {}
    matched = []
    for name, references in auto_map.items():
        references = (references,) if isinstance(references, str) else (references or ())
        if any(
            isinstance(reference, str) and (not architectures or reference.rsplit(".", 1)[-1] in architectures)
            for reference in references
        ):
            matched.append(name)
    # Prefer a task head over an AutoModel entry, independent of dict order.
    for name in matched:
        if name in _VLM_AUTO_NAMES:
            # Preserve the remote auto-map key when that export still exists.
            remote_cls = getattr(transformers, name, None)
            return remote_cls if remote_cls is not None else _get_vlm_auto_class(config)
        if name in ("AutoModelForCausalLM", "AutoModelForTokenClassification", "AutoModelForSequenceClassification"):
            return getattr(transformers, name)
    if matched:
        return transformers.AutoModel

    # Some native VLM heads (e.g. Fuyu) are named ForCausalLM.
    for cls in _vlm_auto_classes():
        if type(config) in getattr(cls, "_model_mapping", {}):
            return cls
    for suffix, name in (
        ("ForTokenClassification", "AutoModelForTokenClassification"),
        ("ForSequenceClassification", "AutoModelForSequenceClassification"),
        ("ForCausalLM", "AutoModelForCausalLM"),
    ):
        if any(suffix in architecture for architecture in architectures):
            return getattr(transformers, name)
    if any(
        suffix in architecture
        for architecture in architectures
        for suffix in ("ForConditionalGeneration", "ForVision2Seq", "ForImageTextToText")
    ):
        return _get_vlm_auto_class(config)
    # Native causal configs can legitimately have architectures=None.
    causal_cls = transformers.AutoModelForCausalLM
    if type(config) in getattr(causal_cls, "_model_mapping", {}):
        return causal_cls
    return transformers.AutoModel


def resolve_lora_target_modules(target_modules):
    """Expand the opt-in MedGemma language-only target; leave other values intact."""
    if isinstance(target_modules, str) and target_modules == "medgemma-language-linear":
        return _MEDGEMMA_LANGUAGE_LINEAR
    return target_modules


def get_text_forward_kwargs(config, input_ids):
    """Gemma3 VLM text inputs need all-text token types, on the input device.

    Gemma3Text/Qwen and other causal heads must not receive this VLM-only kwarg.
    Callers with actual multimodal token types should retain those over this
    text-only default. zeros_like preserves input dtype, shape and device, and
    retains the primitive used by the original single-step MedGemma adapter.
    """
    if getattr(config, "model_type", None) == "gemma3":
        import torch

        return {"token_type_ids": torch.zeros_like(input_ids)}
    return {}
