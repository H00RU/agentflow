# 🚀 GitHub 推送指南

## 当前状态

✅ 代码已修复
✅ Git 已初始化并提交 (921 files, 143,630 lines)
✅ GitHub 仓库已创建: https://github.com/beita6969/agnetworkflow
✅ Remote 已配置
⏳ 等待推送

---

## 方法1: 使用 GitHub CLI (推荐)

### 步骤 1: 打开终端
```bash
cd "/Users/zhangmingda/Desktop/agent worflow"
```

### 步骤 2: 登录 GitHub CLI
```bash
gh auth login
```

选择:
- `GitHub.com`
- `HTTPS`
- `Yes` (authenticate Git with your GitHub credentials)
- `Login with a web browser`

会显示一个验证码，复制它并在浏览器中打开链接授权。

### 步骤 3: 推送代码
```bash
git push -u origin main
```

---

## 方法2: 使用 Personal Access Token

### 步骤 1: 生成 Token
1. 访问: https://github.com/settings/tokens
2. 点击 "Generate new token" → "Generate new token (classic)"
3. 设置:
   - Note: `agnetworkflow-push`
   - Expiration: 选择一个期限
   - 勾选权限: `repo` (全部)
4. 点击 "Generate token"
5. **复制生成的 token（只显示一次！）**

### 步骤 2: 推送代码
```bash
cd "/Users/zhangmingda/Desktop/agent worflow"
git push -u origin main
```

当提示输入:
- **Username**: `beita6969`
- **Password**: 粘贴刚才复制的 token（不是你的 GitHub 密码）

---

## 方法3: 使用 SSH (永久解决方案)

### 步骤 1: 生成 SSH Key
```bash
ssh-keygen -t ed25519 -C "your_email@example.com"
```
按回车使用默认位置，设置密码（可选）

### 步骤 2: 添加 SSH Key 到 GitHub
```bash
# 复制公钥
cat ~/.ssh/id_ed25519.pub
```

访问 https://github.com/settings/keys
- 点击 "New SSH key"
- Title: `MacBook`
- Key: 粘贴刚才复制的内容
- 点击 "Add SSH key"

### 步骤 3: 测试连接
```bash
ssh -T git@github.com
```

### 步骤 4: 修改 remote 并推送
```bash
cd "/Users/zhangmingda/Desktop/agent worflow"
git remote set-url origin git@github.com:beita6969/agnetworkflow.git
git push -u origin main
```

---

## 验证推送成功

推送成功后，访问: https://github.com/beita6969/agnetworkflow

你应该看到:
- 921 个文件
- 最近的提交信息: "Initial commit: AFlow-verl deep integration"
- README、代码文件等

---

## 推送后的下一步

### 1. 在服务器上使用 Git 更新
```bash
ssh root@6.tcp.ngrok.io -p 15577
# 密码: LtgyRHLSCrFm

cd /root/aflow_integration

# 初始化 git (如果还没有)
git init
git remote add origin https://github.com/beita6969/agnetworkflow.git

# 以后更新时
git pull origin main

# 或者直接克隆新副本
cd /root
rm -rf aflow_integration_old
mv aflow_integration aflow_integration_old
git clone https://github.com/beita6969/agnetworkflow.git aflow_integration
```

### 2. 保存配置文件
服务器上的配置文件会在 git pull 时被覆盖，建议备份:
```bash
cp /root/aflow_integration/integration/test_config.yaml ~/test_config.yaml.backup
```

---

## 遇到问题？

### 问题 1: "fatal: could not read Username"
**解决**: 使用方法1或方法2的步骤

### 问题 2: "Host key verification failed"
**解决**: 运行 `ssh-keyscan github.com >> ~/.ssh/known_hosts`

### 问题 3: "Permission denied (publickey)"
**解决**: 检查 SSH key 是否正确添加到 GitHub

### 问题 4: 推送很慢
**原因**: 第一次推送 143K+ 行代码需要时间
**解决**: 耐心等待，或检查网络连接

---

## 快速命令参考

```bash
# 查看状态
git status

# 查看远程仓库
git remote -v

# 查看提交历史
git log --oneline

# 强制推送（谨慎使用）
git push -f origin main

# 查看分支
git branch -a
```

---

**推荐使用方法1（GitHub CLI），最简单！**
