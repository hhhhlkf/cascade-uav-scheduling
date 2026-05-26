/plugin load superpowers

# 项目基本信息

## 远程实验环境 (SSH)

实验代码运行在远程 GPU 服务器上，不在本地执行。

> SSH 连接信息存储在 `ssh-config.local`（仅本地，已 gitignore）。

### 远程工作目录

`/root/autodl-tmp/code/cascade-uav-scheduling/`

### 远程环境初始化（首次）

```bash
# 1. 安装 uv（如未安装）
which uv || curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env  # 或 source $HOME/.cargo/env

# 2. 使用 uv 创建虚拟环境并安装依赖
cd /root/autodl-tmp/code/cascade-uav-scheduling
uv venv --python 3.10
source .venv/bin/activate
uv pip install -r requirements.txt
```

### 远程运行实验

```bash
# 训练 CASCADE
python experiments/run_e1_comparison.py

# 后台运行（断开 SSH 后不中断）
nohup python experiments/run_e1_comparison.py > logs/e1.log 2>&1 &

# 使用 tmux（推荐）
tmux new -s cascade
python experiments/run_e1_comparison.py
# Ctrl+B, D 断开会话
# tmux attach -t cascade 重新连接
```

## 修改后同步要求

每次完成代码、配置、测试或文档修改后，都需要执行以下同步流程：

1. 本地验证优先运行：

```bash
python -m unittest discover -s tests
python experiments/smoke_env.py
```

2. 检查待提交文件，避免提交本地隐私文件、大文件和实验输出：

```bash
git status --short --ignored
```

3. 将本次修改提交并推送到 GitHub：

```bash
git add <changed-source-files>
git commit -m "<message>"
git push origin master
```

4. 推送成功后，登录远程实验服务器并拉取最新版本：

```bash
cd /root/autodl-tmp/code/cascade-uav-scheduling
git pull --ff-only origin master
```

如果远程服务器存在未提交本地改动，必须先确认来源与用途，不能直接覆盖。
