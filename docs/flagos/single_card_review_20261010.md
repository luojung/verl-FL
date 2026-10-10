# FlagOS 单卡 Qwen / MedGemma 适配审查记录

日期：2026-10-10。基线：官方 `flagos-ai/verl-FL` 提交
`45068d9ef924ade2485dd6137e955be6e3fb0042`，**不是对官方最新分支的兼容承诺**。
历史 MedGemma 实验补丁已迁入此基线，再经过四个分工审查 agent 和独立复审。

## 1. 任务边界与成果含义

- 面向 verl 的 **LLM / VLM 文本路径后训练**，不承诺任意科学模型接入。
- Qwen 因果语言模型与 MedGemma/Gemma3 条件生成模型共用局部兼容层；没有全局修改 Transformers 导出。
- 本轮重点是历史单卡适配的代码安全性、可重复性和错误可观察性。
- 共享 HF rollout 仅支持 **单 rank、legacy FSDP1、根包装 NO_SHARD / use_orig_params=True、文本输入**。
  这不是分布式 HF 推理后端，不支持外部异步权重同步。
- 不承诺 MedGemma 图像训练、FSDP2 HF rollout、多卡 GRPO、张量/流水线并行、模型质量提升或性能达标。
- 原框架的多卡 SFT 与 vLLM/SGLang 路径保留；“没有主动移除”不等于本轮已经实测。
- 历史完整权重单 DCU 运行结果与本次修订版的验证是两份证据，不能互相替代。

## 2. 模型与环境

历史任务使用 MedGemma 4B 指令模型的文本输入、BF16、LoRA、小批量 SFT / HF GRPO。
本轮修复不是将 MedGemma 换成 Qwen：两者通过配置类型和 architecture 自动选择不同 AutoModel。
Qwen 示例应选择与安装的 Transformers 兼容且单卡容得下的 **因果语言模型**；不能将任意 Qwen 多模态/MoE 变体视为验收通过。

真实 CPU 预检环境：Python 3.10.20、PyTorch 2.10.0+cpu、Transformers 5.5.0、PEFT 0.19.1、
safetensors 0.8.0。这个版本组合不是完整的支持矩阵。
本地 isolated 测试环境没有 torch / transformers / peft / ray，主要依赖 mocks 和 AST 提取。
运行完整训练需另外准备匹配的 Torch-FL / 厂商 runtime、Ray、TensorDict、Hydra、torchdata 等。
不在示例里自动安装或更改共享环境。

## 3. 做了哪些适配与修复

| 范围 | 实现位置 | 修复 / 设计 |
| --- | --- | --- |
| 平台选择 | `verl/plugin/platform/platform_flagos.py`、`platform_manager.py` | 显式 FlagOS 初始化失败直接报错；可选探测不破坏原后端；只缓存成功初始化；不重复初始化 |
| Ray 调度与可见卡 | `verl/single_controller/ray/base.py` | FlagOS 请求 Ray `GPU`；沿用 Worker 对 HIP/ROCR 的规范化；契约统一为 `CUDA_VISIBLE_DEVICES`；不支持自定义可见卡变量 |
| 观测 | `verl/utils/flops_counter.py`、`utils/profiler/performance.py` | 不支持的显存计数显示 `N/A`；未知算力传播 `NaN`；不冒充成功的设备 profiling |
| 模型兼容 | `verl/utils/model_compat.py`及加载调用点 | 局部新旧 VLM AutoModel 选择；Gemma3 文本 token types；专用语言层 LoRA regex；保持普通 Qwen targets |
| SFT | `verl/trainer/fsdp_sft_trainer.py`、`utils/dataset/sft_dataset.py` | 真实 logits 词表维度；单行 pandas 索引；安全 DataLoader 默认；eager 可配置；移除强制 `os._exit`，暴露清理错误 |
| HF rollout | `verl/workers/rollout/hf_rollout.py` | 直接 chat tokenization；左 padding；长度覆盖一致；bounded microbatch split；填充尾部 mask；不传静态 decode position IDs |
| HF/FSDP 生命周期 | `verl/workers/fsdp_workers.py` | 单卡根 FSDP / NO_SHARD；所有共享 HF actor 强制原始参数，支持混合冻结 LoRA；参数载入/卸载、独立 RNG、各子模块模式在 finally 恢复；去除私有 lazy-init reset |
| PPO 路由 | `verl/trainer/main_ppo.py`、`trainer/ppo/ray_trainer.py` | HF 使用 legacy 同步 worker，跳过 async agent manager；明确拒绝不支持的 engine / reward-loop 配置 |
| LoRA 导出 | `verl/utils/checkpoint/lora_adapter.py`、`utils/fsdp_utils.py`、actor checkpoint 调用点 | 公共 full-state API 收集；PEFT 负责过滤；空 adapter 报错；独立 CPU tensor；regex 保留；全 rank 错误协议；rank0 写盘失败不能伪成功 |

### 必须注意的行为变化

1. 显式 `VERL_PLATFORM=flagos` 不再静默回退 CPU。
2. `FLAGOS_VISIBLE_DEVICES_ENV` 不得改成 HIP/ROCR。不要把 Ray 分配的卡号硬改成 `0`，也不要在 runtime 已初始化后修改可见卡。
3. `rollout.name=hf` 需要 `reward_model.use_reward_loop=false` 和 legacy FSDP1。
   schema 仍使用 `rollout.mode=async`，但 trainer 按 **backend name** 选择同步 HF 路径。
4. `medgemma-language-linear` 是专用 regex 入口，普通 Qwen 不使用这一入口。
5. GRPO actor checkpoint 中的 adapter 导出进行了加强；SFT 仍沿用原框架 FSDP checkpoint manager，不能说 SFT 也已经变成 adapter-only 导出。
6. adapter 导出会先收集完整 base state，可能增加峰值内存。
   分布式错误协议要求默认进程组成员以相同顺序调用；不保证任意子进程组或卡死 collective 恢复。
7. adapter 的两个文件采用独立替换，**不是跨文件事务**。失败会报错，但可能存在部分发布；不要忽略失败直接消费输出。

## 4. 本轮验证与验收状态

| 验证 | 状态 / 范围 |
| --- | --- |
| 合并 isolated/mock 回归 | **312 passed**；平台、Ray Worker 可见卡逻辑、遥测、模型兼容、SFT、rollout、路由、导出、worker 生命周期 |
| Ruff lint / format | 修改的 Python 文件通过；后续新增 runtime / example 文件另行验证 |
| `git diff --check` | 通过；Windows LF/CRLF 提示不代表内容错误 |
| 真 CPU 模型 / PEFT / FSDP 冒烟 | **4 cases passed, exit 0**；普通 / 根 FSDP1 的微型 Qwen2、Gemma3 条件生成文本路径 |
| 真 CPU HF rollout | **2 cases passed, exit 0**；微型 Qwen2 / Gemma3 causal text，更新前后共四次 generation；修复生成配置 warning 后最终源码复测通过 |
| 示例 Hydra composition | **4 variants passed**；真实 Hydra 组合，不包含 dataclass 实例化或完整 trainer 启动 |
| 修订版完整 MedGemma 单 DCU SFT / GRPO | **本轮未重跑**，历史成果不能充当修订版验收 |
| 修订版完整 Qwen 单 DCU SFT / GRPO | **本轮未完成** |
| 多卡 SFT / GRPO | **本轮未验收** |

### 真实 CPU 验证记录

- Run `20261010-181728-054-verl-real-cpu-smoke-v2`：2026-10-10，Slurm 分配的计算节点，
  FP32 / eager / 单 rank Gloo；Qwen2 与 Gemma3ForConditionalGeneration 的普通模型和根 FSDP1 共四例通过。
- 每例均验证有限 loss、非零 LoRA 梯度、真实参数变化、冻结 base 不变、非空 safetensors，
  真实 PEFT config / adapter 复载；最大 logits 复载差异全部为 **0**。
- Qwen2 adapter 导出 8 个 tensor，Gemma3 语言层 adapter 导出 28 个 tensor。
- 测试直接加载生产 `model_compat` / `lora_adapter` 实现；`fsdp_version` 从生产源码提取，
  绑定真实 FSDP 类。**不验证整个 verl 模块 import、完整 trainer 或 pretrained MedGemma 权重**。
- 首次 run `20261010-181335-173-verl-real-cpu-smoke` exit 1：Torch-FL 自动加载修改了 HIP metadata，
  触发纯 CPU 环境检查，模型训练尚未开始。重试在新进程关闭 backend extension autoload，
  不修改共享环境。该失败未计为通过，现场单独保留。
- HF run `20261010-181940-490-verl-real-hf-cpu-smoke` exit 0：真实 TensorDict / DataProto / HF 模型 / FSDP，
  验证 actor 更新前后共享 generation、response-length 覆盖、mask、position bookkeeping 和模式恢复。
  使用 Gemma3ForCausalLM 文本模型，不代表完整 MedGemma VLM rollout；生产定义源加载，不覆盖完整 import 栈。
- 最终 run `20261010-182456-672-verl-final-cpu-regressio` exit 0：重新验证四个训练/adapter 测试、
  两个 HF 测试，以及四套示例的真实 Hydra composition。生成相关设置已统一放入 GenerationConfig，
  不再出现混用 generation kwargs 的弃用 warning。未实例化所有 config dataclass，也未启动完整 trainer。
- 最终测试源码 archive SHA-256：`c24dd26dfe0f96316a18853d266dec1daf228cc1836cb6c3074e0622502ba5ed`。
- 原始日志和内部服务器路径在本地交付包保留，不提交到公开仓库。

```bash
python tests/runtime/single_card_cpu_smoke.py --require-fsdp
python tests/runtime/hf_rollout_cpu_smoke.py
```

CPU 测试关闭 `TORCH_DEVICE_BACKEND_AUTOLOAD`；这只是纯 CPU 对照，不是 DCU 的配置建议。

isolated 测试复现：

```bash
python -m pytest -q \
  tests/plugin/test_flagos_platform.py \
  tests/plugin/test_flagos_ray.py \
  tests/plugin/test_flagos_telemetry.py \
  tests/trainer/test_hf_routing_single_card.py \
  tests/trainer/test_sft_compat_isolated.py \
  tests/utils/test_lora_export_contract.py \
  tests/utils/test_model_compat_isolated.py \
  tests/workers/test_fsdp_worker_flagos_integration.py \
  tests/workers/test_hf_rollout_single_card.py
```

这些测试不是完整项目 test suite，也不能证明真实 NCCL/RCCL/FlagOS collective 正确。

## 5. 复现入口

- `examples/flagos_single_card/prepare_smoke_data.py`：生成小样本测试数据，不代表业务训练集。
- `examples/flagos_single_card/run_sft.sh`：参数化单卡 SFT。
- `examples/flagos_single_card/run_grpo_hf.sh`：参数化单卡 HF GRPO。
- `examples/flagos_single_card/smoke_reward.py`：仅用于流程验证，不能作为医学质量或临床安全评估。
- `tests/runtime/single_card_cpu_smoke.py`：不下载模型权重的真实 CPU 微型模型与 adapter 测试。
- `tests/runtime/hf_rollout_cpu_smoke.py`：根 FSDP actor 更新前后共享 HF generation 测试。

详细环境变量与调用方式见示例目录 README。必须在已分配的计算资源中执行，
由平台已有的调度/可视化规范启动，不能在登录节点运行完整 4B 模型。
只在输出、参数变化、checkpoint 复载和进程退出全部检查后才能判定训练成功。

## 6. 底层运行逻辑与原理分析

本次审查追踪的是 **文本 token → 训练/生成 → adapter 输出**：

```text
parquet / chat messages
 → SFTDataset / HFRollout tokenization
 → input_ids + attention_mask（Gemma3 文本路径补 token_type_ids）
 → 模型语言层：embedding、attention、linear、logits
 → SFT cross-entropy 或 GRPO log-prob / advantage / policy loss
 → Autograd → LoRA 梯度 → optimizer 更新
 → FSDP full-state API → PEFT adapter 过滤
 → detach / CPU contiguous clone → safetensors + JSON
```

共享 HF generation 不向另一个 engine 传送 state dict：直接复用 actor，因此不能跳过
参数卸载恢复、RNG 切换与 train/eval 恢复。根 FSDP 包装不对其内部未受管的 decode 子模块
进行多 rank 分片；拒绝 nested/sharded HF 比继续操作私有 `_reset_lazy_init` 更可审查。

从 Python 到设备的链路必须分别判断证据级别：

- **静态/isolated 已验证**：选择 AutoModel、tokenization、mask、生命周期、checkpoint 错误传播。
- **真实 CPU**：runtime 冒烟使用真实 eager forward/backward 和公开 FSDP/PEFT API；日志记录具体 ATen schema。
  CPU dispatcher 路径不能推导 FlagOS/RCCL 的正确性。真实日志显示
  `aten::embedding` 为 CompositeExplicitAutograd 注册，`aten::mm` / `bmm`、
  `_log_softmax`、`nll_loss_forward/backward` 和 `add_.Tensor` 有 CPU 注册；
  输入 dispatch keys 包含 CPU / AutogradCPU。注册查询不是逐 kernel profiler 证据。
- **FlagOS DCU 待复验**：设备注册/PrivateUse1 → Torch-FL 路由 → DAS boxing 或 FlagGems
  → 厂商 kernel，需拿修订版的实际 dispatcher/route 日志和算子输入复验。
  本轮不能指定某个实际 DCU kernel 已命中，也没有新的 DCU 性能数字。
- **编译/融合**：示例采用 eager 并关闭 actor torch compile，本轮没有测量编译、融合或 FlagTree 收益。
  一个 ATen `linear`/`mm` 名称不能当成“已融合”证据。
- **通信**：单 rank NO_SHARD 不等于多 rank 参数 all-gather / gradient reduce-scatter 验证；
  多卡验收必须另查默认组、设备组、可见卡与通信 backend，并覆盖异常退出。

责任边界：本轮修复是 verl 集成层；如果下一阶段出现 PrivateUse1 注册/boxing/通信失败，
需归因 Torch-FL/runtime；具体公共算子失败再分离 FlagGems 与 boxing 对照。
不能把框架路由或 FSDP 生命周期错误直接交给算子库“补 kernel”。

## 7. 下一阶段目标与交付验收

1. **单卡修订版复验**：Qwen 与 MedGemma 各一套 SFT / GRPO，一次真实参数更新、有限 loss / log-prob、
   非空 checkpoint、adapter 或合并权重复载、明确退出码，无强制退出伪成功。
2. **多卡 SFT**：先 2 卡，再目标卡数；验证 rank 独立映射、loss/gradient、checkpoint save/load、
   单卡对照及正常 destroy process group。
3. **多卡 GRPO**：另行设计 rollout backend 与权重同步；不能仅把当前脚本的卡数改大。
   选用适配过的 vLLM/SGLang/其他后端，或新增支持的分布式 HF 实现，并验证 group sampling 和梯度语义。
4. **性能与质量**：扩大数据和步数后才谈收敛；分别记录吞吐、显存、通信和数值偏差，
   profiling 不支持时诚实标记缺失。
5. **交付物**：源码 commit、版本/环境锁定、参数化脚本、测试数据 schema、原始日志、产物校验、
   复载对照、问题清单与边界说明。不要用“top20 全部跑通”替代架构/任务选择与逐模型证据。
