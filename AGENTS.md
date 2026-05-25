# AGENTS.md

## 回答语言

以中文为主，除非用户明确要求使用其他语言。

## 项目背景

本仓库用于实现 CASCADE 洪涝灾害异构无人机集群协同任务调度实验。当前优先级是按照 `doc/实验计划_CASCADE_仿真环境与算法框架.md` 推进纯 RL 版本，不涉及 LLM 调度决策。

## 开发原则

- 优先按实验计划 Phase 1 搭建仿真环境核心：`ScenarioGenerator`、`UAVSimulator`、`MeshNetworkSimulator`、`TaskManager`、`action_mask`、`reward`、`CASCADEEnv`。
- 仿真环境需保持 Gymnasium 风格接口：`reset()`、`step(action)`、`get_action_mask()`。
- 配置优先放在 `configs/` 下，场景参数使用 YAML 管理。
- Python 代码尽量保持模块化，公共数据结构放在 `src/utils/types.py`。
- 每次改动后优先运行：

```bash
python -m unittest discover -s tests
python experiments/smoke_env.py
```

## 远程实验环境

实验代码运行在远程 GPU 服务器上，不在本地执行长训练。

远程工作目录：

```bash
/root/autodl-tmp/code/cascade-uav-scheduling/
```

远程环境初始化：

```bash
which uv || curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

cd /root/autodl-tmp/code/cascade-uav-scheduling
uv venv --python 3.10
source .venv/bin/activate
uv pip install -r requirements.txt
```

## 隐私与本地文件

- SSH 连接信息存储在 `ssh-config.local`，属于本地文件，不提交。
- `.claude/` 属于本地助手配置，不提交。
- 不提交大模型权重、实验输出、日志、checkpoint、PDF 等大文件。

## Git 注意事项

- `src/env/` 和 `configs/env/` 是项目源码与配置，不应被虚拟环境忽略规则误伤。
- 提交前检查 `git status --short --ignored`，确认只跟踪应该入库的源码、配置、测试和文档。
