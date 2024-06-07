# 标题五个字

今天在实习中遇到一个问题，在我提交代码到远程分支并合并到主分支时，pl发现我的代码修改了部分不应该修改的文件。

于是乎学习下如何使用git撤销这些未修改的文件

```
# 查看提交记录并且找到你希望恢复到哪一次commit的ID
git log
 
git reset <commit_id> -- <path_to_file>

git add <path_to_file>

git commit --amend --no-edit

git push origin <your_branch> -f
```
