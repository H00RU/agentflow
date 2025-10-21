#!/bin/bash

echo "🚀 Pushing code to GitHub..."
echo ""

cd "/Users/zhangmingda/Desktop/agent worflow"

# 确保使用HTTPS URL
git remote set-url origin https://github.com/beita6969/agnetworkflow.git

echo "Repository: https://github.com/beita6969/agnetworkflow"
echo ""
echo "Pushing to GitHub..."
echo ""

# 推送代码
git push -u origin main

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ 成功推送到GitHub!"
    echo "📁 查看你的仓库: https://github.com/beita6969/agnetworkflow"
else
    echo ""
    echo "❌ 推送失败"
    echo ""
    echo "请尝试以下步骤："
    echo "1. 确保你已经登录GitHub"
    echo "2. 生成Personal Access Token: https://github.com/settings/tokens"
    echo "3. 再次运行此脚本，使用用户名和token作为密码"
fi
