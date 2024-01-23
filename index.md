---
title: 联机托管说明
permalink: index.html
layout: home
---

# Azure Databricks 练习

这些练习旨在支持 Microsoft Learn 上的以下培训内容：

- [使用 Azure Databricks 执行数据工程](https://learn.microsoft.com/training/paths/data-engineer-azure-databricks/)
- [使用 Azure Databricks 实现机器学习](https://learn.microsoft.com/training/paths/build-operate-machine-learning-solutions-azure-databricks/)

需要一个你在其中具有管理权限的 Azure 订阅才能完成这些练习。

{% assign exercises = site.pages | where_exp:"page", "page.url contains '/Instructions/Exercises'" %} {% for activity in exercises  %}
- [{{ activity.lab.title }}]({{ site.github.url }}{{ activity.url }}) | {% endfor %}