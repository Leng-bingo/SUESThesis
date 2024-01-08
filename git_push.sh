#!/bin/bash

# 确保脚本在发生任何错误时终止
set -e

# 检查是否提供了提交消息
if [ -z "$1" ]; then
  echo "Usage: ./git_push.sh 'Your commit message'"
  exit 1
fi

# 提取提交消息
COMMIT_MESSAGE=$1

# 执行Git命令
# 拉取最新的远程仓库
# git pull origin main

# 添加所有变更的文件
git add .

# 提交变更
git commit -m "$COMMIT_MESSAGE"

# 推送到远程仓库
git push -u origin main 

echo "Changes have been pushed to GitHub with message: $COMMIT_MESSAGE"
