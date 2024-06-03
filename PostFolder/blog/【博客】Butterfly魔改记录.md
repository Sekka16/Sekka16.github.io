---
title: 【博客】Butterfly魔改记录
date: 2024-04-02 17:56:32
tags:
description: 记录所作的魔改防止哪天忘了
---
# 基于Butterfly主题的魔改记录

## 关于背景

首先是背景图片

在`_config.butterfly.yml`中修改：

```yml
# 主页横幅
index_img: ./img/bk.jpg

# 网站的背景图片
# 文档中有描写本地路径需要加上根目录，但是尝试之后并没有成功
background: url(https://raw.githubusercontent.com/Sekka16/Sekka16.github.io/main/source/images/bk.jpg)

# 保证文章的默认横幅是透明的
# If the banner of page not setting, it will show the top_img
default_top_img: transparent

# 页脚同理
# Footer Background
footer_bg: transparent

```

透明度设置

butterfly主题的`source/css`文件夹下新建`transparent.css`，内容如下
```css
/* 文章页背景 */
.layout_post>#post {
    /* 以下代表透明度为0.7 可以自行修改*/
    background: rgba(255, 255, 255, .7);
}

/* 所有页面背景 */
#aside_content .card-widget,
#recent-posts>.recent-post-item,
.layout_page>div:first-child:not(.recent-posts),
.layout_post>#page,
.layout_post>#post,
.read-mode .layout_post>#post {
    /* 以下代表透明度为0.7 */
    background: rgba(255, 255, 255, .7);
}

/*侧边卡片的透明度 */
:root {
    --card-bg: rgba(255, 255, 255, .7);
}

/* 页脚透明 */
#footer {
    /* 以下代表透明度为0.7 */
    background: rgba(255, 255, 255, .0);
}
```

之后在`_config.butterfly.yml`中修改：
```yml
# Inject
# Insert the code to head (before '</head>' tag) and the bottom (before '</body>' tag)
# 插入代码到头部 </head> 之前 和 底部 </body> 之前
inject:
  head:
    # - <link rel="stylesheet" href="/xxx.css">
    - <link rel="stylesheet" href="/css/transparent.css">
```