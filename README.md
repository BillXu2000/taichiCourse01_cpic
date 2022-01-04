# 让我试试刀吧！

这次作业是 https://yuanming.taichi.graphics/publication/2018-mlsmpm/ 这篇论文的一个简单复现。

文中 cpic 的方法支持了刚体与 mpm 粒子的双向耦合，同时也部分支持了用很薄的刀刃对 mpm 粒子进行切割，本次作业的重心更多放在了切割上（其实是双向耦合还没调出来）。

祝大家切得开心。

## 成功效果展示

![fractal demo](./data/cpic.gif)

## 运行方式
`python3 main.py`

## future works

刚体移动时经常会发生少量粒子穿透的现象，之后有空再试着修一修。
