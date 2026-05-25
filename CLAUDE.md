/plugin load superpowers

# 项目基本信息

## 远程实验环境 (SSH)

实验代码运行在远程 GPU 服务器上，不在本地执行。

### SSH 连接指令

```bash
# SSH 连接 (使用以下模板，填入实际信息后取消注释)
# ssh -p <SSH端口> <用户名>@<服务器IP>

# 示例 (AutoDL):
# ssh -p 12345 root@region-1.autodl.com
```

### 连接信息（待填写）

| 项目 | 值 |
|------|-----|
| **SSH 地址** | `<服务器IP或域名>` |
| **SSH 端口** | `<端口号>` |
| **用户名** | `<用户名，通常为 root>` |
| **密码** | `<登录密码>` |
| **工作目录** | `~/autodl-tmp/code/scheduling-rl/` |

### 免密登录配置 (可选)

```bash
# 本地生成密钥（如已有则跳过）
# ssh-keygen -t rsa -b 4096

# 上传公钥到远程服务器
# ssh-copy-id -p <SSH端口> <用户名>@<服务器IP>
```

### 远程环境初始化

```bash
# SSH 登录后
cd ~/autodl-tmp/code/
git clone <本仓库地址> scheduling-rl
cd scheduling-rl
pip install -r requirements.txt
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
