---
title: 联机托管说明
permalink: index.html
layout: home
---

# Azure Databricks 练习

这些练习旨在支持 Microsoft Learn 上的以下培训内容：

- [使用 Azure Databricks 实现数据湖屋分析解决方案](https://learn.microsoft.com/training/paths/data-engineer-azure-databricks/)
- [使用 Azure Databricks 实现机器学习解决方案](https://learn.microsoft.com/training/paths/build-operate-machine-learning-solutions-azure-databricks/)
- [使用 Azure Databricks 实现数据工程解决方案](https://learn.microsoft.com/en-us/training/paths/azure-databricks-data-engineer/)
- [使用 Azure Databricks 实现生成式 AI 工程](https://learn.microsoft.com/en-us/training/paths/implement-generative-ai-engineering-azure-databricks/)

需要一个你在其中具有管理权限的 Azure 订阅才能完成这些练习。

{% assign exercises = site.pages | where_exp:"page", "page.url contains '/Instructions'" %} {% for activity in exercises  %}
- [{{ activity.lab.title }}]({{ site.github.url }}{{ activity.url }}) | {% endfor %}
