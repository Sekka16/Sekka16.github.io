---
title: [git] git提交到错误分支的处理办法 
date: 2024-09-12 00:00:00 
tags:
- git
categories:
- [工具, git]
description: 
---

# [git] git提交到错误分支的处理方法

## 不同文件提交到不同分支

我们假设有这样一个场景：
  - `branchA`: 对`a.txt`进行修改。
  - `branchB`: 对`b.txt`进行修改。

此时你兴致冲冲解决了`branchB`的活儿，却交到了`branchA`，而且你使用了`git commit --amend --no-edit`，并且已经push到了远程仓库。

至于为什么会这么做，问就是改完代码忘记改文档被reviewer叫回来了。

问题来了：怎么将`b.txt`的修改提交至对应分支而不影响另一个分支？

首先重置当前分支的 HEAD 到上一次提交，在 Git 中，HEAD 是一个特殊的指针，指向当前分支的最新提交。HEAD^ 是 HEAD 的父提交，表示当前分支的最新提交的上一次提交。


```bash
git reset HEAD^

```

第二步，将指定的文件保存到stash中

```bash
git stash push b.txt
```

此时a.txt就可以正常提交了
```bash
git add a.txt
git commit -m "Modify a.txt by branchA."
git push origin branchA
```

第三步，切换到`branchB`，取出b.txt，然后add, commit, push三连击
```bash
git checkout branchB
git stash pop b.txt
...
```


