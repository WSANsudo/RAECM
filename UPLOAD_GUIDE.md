# GitHub 上传指南 / GitHub Upload Guide

## 中文版

### 准备工作

1. **安装 Git**
   - Windows: 下载 [Git for Windows](https://git-scm.com/download/win)
   - 验证安装: `git --version`

2. **配置 Git（首次使用）**
   ```bash
   git config --global user.name "你的名字"
   git config --global user.email "你的邮箱"
   ```

3. **创建 GitHub 账号**
   - 访问 [github.com](https://github.com) 注册账号

### 上传步骤

#### 方法一：通过命令行（推荐）

1. **在 GitHub 上创建新仓库**
   - 登录 GitHub
   - 点击右上角 "+" → "New repository"
   - 仓库名称: `6Analyst` 或 `RAECM`
   - 描述: `Evidence-Centric Multi-Agent Framework for Router Asset Identification`
   - 选择 Public 或 Private
   - **不要**勾选 "Initialize with README"（我们已经有了）
   - 点击 "Create repository"

2. **在本地初始化 Git 仓库**
   ```bash
   cd 6Analyst-master
   git init
   git add .
   git commit -m "Initial commit: RAECM framework"
   ```

3. **连接到 GitHub 并推送**
   ```bash
   # 替换 YOUR_USERNAME 和 YOUR_REPO_NAME
   git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
   git branch -M main
   git push -u origin main
   ```

4. **输入 GitHub 凭据**
   - 用户名: 你的 GitHub 用户名
   - 密码: 使用 Personal Access Token（不是账号密码）
   
   **获取 Personal Access Token:**
   - GitHub → Settings → Developer settings → Personal access tokens → Tokens (classic)
   - Generate new token → 勾选 `repo` 权限 → Generate token
   - 复制 token（只显示一次！）

#### 方法二：通过 GitHub Desktop（简单）

1. **下载 GitHub Desktop**
   - 访问 [desktop.github.com](https://desktop.github.com)
   - 下载并安装

2. **使用 GitHub Desktop**
   - 打开 GitHub Desktop
   - File → Add Local Repository → 选择 `6Analyst-master` 文件夹
   - 点击 "Publish repository"
   - 填写仓库名称和描述
   - 选择 Public 或 Private
   - 点击 "Publish repository"

### 重要提醒

⚠️ **上传前检查清单:**

- [ ] 确保没有 API 密钥在代码中（已通过 .gitignore 排除）
- [ ] 检查是否有敏感数据（IP地址、密码等）
- [ ] 确认 README.md 内容正确
- [ ] 验证 requirements.txt 完整

⚠️ **不要上传的内容:**
- API 密钥文件
- 大型模型文件（>100MB）
- 个人数据或敏感信息
- 临时文件和缓存

### 后续更新

当你修改代码后，更新到 GitHub:

```bash
cd 6Analyst-master
git add .
git commit -m "描述你的修改"
git push
```

---

## English Version

### Prerequisites

1. **Install Git**
   - Windows: Download [Git for Windows](https://git-scm.com/download/win)
   - Verify: `git --version`

2. **Configure Git (first time)**
   ```bash
   git config --global user.name "Your Name"
   git config --global user.email "your.email@example.com"
   ```

3. **Create GitHub Account**
   - Visit [github.com](https://github.com) and sign up

### Upload Steps

#### Method 1: Command Line (Recommended)

1. **Create New Repository on GitHub**
   - Log in to GitHub
   - Click "+" in top-right → "New repository"
   - Repository name: `6Analyst` or `RAECM`
   - Description: `Evidence-Centric Multi-Agent Framework for Router Asset Identification`
   - Choose Public or Private
   - **Do NOT** check "Initialize with README" (we already have one)
   - Click "Create repository"

2. **Initialize Local Git Repository**
   ```bash
   cd 6Analyst-master
   git init
   git add .
   git commit -m "Initial commit: RAECM framework"
   ```

3. **Connect to GitHub and Push**
   ```bash
   # Replace YOUR_USERNAME and YOUR_REPO_NAME
   git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
   git branch -M main
   git push -u origin main
   ```

4. **Enter GitHub Credentials**
   - Username: Your GitHub username
   - Password: Use Personal Access Token (not account password)
   
   **Get Personal Access Token:**
   - GitHub → Settings → Developer settings → Personal access tokens → Tokens (classic)
   - Generate new token → Check `repo` scope → Generate token
   - Copy token (shown only once!)

#### Method 2: GitHub Desktop (Easy)

1. **Download GitHub Desktop**
   - Visit [desktop.github.com](https://desktop.github.com)
   - Download and install

2. **Use GitHub Desktop**
   - Open GitHub Desktop
   - File → Add Local Repository → Select `6Analyst-master` folder
   - Click "Publish repository"
   - Fill in repository name and description
   - Choose Public or Private
   - Click "Publish repository"

### Important Reminders

⚠️ **Pre-upload Checklist:**

- [ ] No API keys in code (excluded via .gitignore)
- [ ] No sensitive data (IP addresses, passwords, etc.)
- [ ] README.md content is correct
- [ ] requirements.txt is complete

⚠️ **Do NOT Upload:**
- API key files
- Large model files (>100MB)
- Personal or sensitive data
- Temporary files and caches

### Subsequent Updates

After modifying code, update to GitHub:

```bash
cd 6Analyst-master
git add .
git commit -m "Describe your changes"
git push
```

---

## 常见问题 / FAQ

### Q: 文件太大无法上传怎么办？
**A:** GitHub 单个文件限制 100MB。对于大文件：
- 使用 Git LFS (Large File Storage)
- 或将大文件放在其他地方（如 Google Drive）并在 README 中提供链接

### Q: 如何设置仓库为私有？
**A:** 在创建仓库时选择 "Private"，或在仓库 Settings → Danger Zone → Change visibility

### Q: 忘记添加 .gitignore 怎么办？
**A:** 
```bash
git rm -r --cached .
git add .
git commit -m "Update .gitignore"
git push
```

### Q: 如何删除已上传的敏感文件？
**A:** 使用 BFG Repo-Cleaner 或 git filter-branch（复杂，建议查阅文档）

---

## 推荐的仓库设置

上传后，在 GitHub 仓库中设置：

1. **添加 Topics（标签）**
   - Settings → Topics
   - 添加: `router-identification`, `llm`, `multi-agent`, `network-security`, `asset-discovery`

2. **启用 Issues**
   - Settings → Features → Issues ✓

3. **添加 Description**
   - 在仓库首页点击 "Add description"
   - 填写: "Evidence-Centric Multi-Agent Framework for Router Asset Identification"

4. **设置 About**
   - 添加网站链接（如果有）
   - 添加相关论文链接

---

## 需要帮助？

- Git 官方文档: https://git-scm.com/doc
- GitHub 帮助: https://docs.github.com
- Git 教程: https://www.atlassian.com/git/tutorials
