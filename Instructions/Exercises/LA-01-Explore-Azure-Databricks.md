---
lab:
  title: 了解 Azure Databricks
---

# 了解 Azure Databricks

Azure Databricks 是基于 Microsoft Azure 的常用开源 Databricks 平台的一个版本。

Azure Databricks *工作区*为管理 Azure 上的 Databricks 群集、数据和资源提供了一个中心点。

在本练习中，你将预配 Azure Databricks 工作区并探索其一些核心功能。 

完成此练习大约需要 **20** 分钟。

> **备注**：Azure Databricks 用户界面可能会不断改进。 自编写本练习中的说明以来，用户界面可能已更改。

## 预配 Azure Databricks 工作区

> **提示**：如果你已有 Azure Databricks 工作区，则可以跳过此过程并使用现有工作区。

1. 登录到 Azure 门户，地址为 ****。
2. 请使用以下设置创建 **Azure Databricks** 资源：
    - 订阅****：*选择 Azure 订阅*
    - **资源组**：*创建名为 `msl-xxxxxxx`（其中“xxxxxxx”是唯一值）的新资源组*
    - **工作区名称**：`databricks-xxxxxxx`*（其中“xxxxxxx”是资源组名称中使用的值）*
    - **区域**：*选择任何可用区域*
    - **定价层**：*高级*或*试用版*
    - **受管理资源组名称**：`databricks-xxxxxxx-managed`*（其中“xxxxxxx”是资源组名称中使用的值）*

    ![Azure 门户中“创建 Azure Databricks 工作区”页面的屏幕截图。](./images/create-databricks.png)

3. 选择“**查看 + 创建**”，然后等待部署完成。 然后转到资源并启动工作区。

## 创建群集

Azure Databricks 是一个分布式处理平台，可使用 Apache Spark 群集在多个节点上并行处理数据。 每个群集由一个用于协调工作的驱动程序节点和多个用于执行处理任务的工作器节点组成。 在本练习中，将创建一个*单节点*群集，以最大程度地减少实验室环境中使用的计算资源（在实验室环境中，资源可能会受到限制）。 在生产环境中，通常会创建具有多个工作器节点的群集。

> **提示**：如果 Azure Databricks 工作区中已有一个具有 13.3 LTS ML 或更高运行时版本的群集，则可以使用它来完成此练习并跳过此过程。

1. 在 Azure 门户中，浏览到 **msl-*xxxxxxx*** 资源组（或包含现有 Azure Databricks 工作区的资源组）并选择 Azure Databricks 服务资源。
1. 在工作区的“概述”**** 页中，使用“启动工作区”**** 按钮在新的浏览器标签页中打开 Azure Databricks 工作区；请在出现提示时登录。

    > 提示：使用 Databricks 工作区门户时，可能会显示各种提示和通知。 消除这些内容，并按照提供的说明完成本练习中的任务。

1. 在左侧边栏中，选择“**(+)新建**”任务，然后选择“**群集**”（可能需要查看“**更多**”子菜单）。
1. 在“新建群集”页中，使用以下设置创建新群集：
    - 群集名称：用户名的群集（默认群集名称）
    - **策略**：非受限
    - 群集模式：单节点
    - 访问模式：单用户（选择你的用户帐户）
    - **Databricks 运行时版本**：13.3 LTS（Spark 3.4.1、Scala 2.12）或更高版本
    - 使用 Photon 加速：已选择
    - **节点类型**：Standard_D4ds_v5
    - 在处于不活动状态 20 分钟后终止**********

1. 等待群集创建完成。 这可能需要一到两分钟时间。

> 注意：如果群集无法启动，则订阅在预配 Azure Databricks 工作区的区域中的配额可能不足。 请参阅 [CPU 内核限制阻止创建群集](https://docs.microsoft.com/azure/databricks/kb/clusters/azure-core-limit)，了解详细信息。 如果发生这种情况，可以尝试删除工作区，并在其他区域创建新工作区。

## 使用 Spark 分析数据

与许多 Spark 环境一样，Databricks 支持使用笔记本来合并笔记和交互式代码单元格，可用于探索数据。

1. 将 [**products.csv**](https://raw.githubusercontent.com/MicrosoftLearning/mslearn-databricks/main/data/products.csv) 文件从 `https://raw.githubusercontent.com/MicrosoftLearning/mslearn-databricks/main/data/products.csv` 下载到本地计算机，并将其另存为 **products.csv**。
1. 在边栏的“**(+)新建**”链接菜单中，选择“**添加或上传数据**”。
1. 选择“**创建或修改表**”，并将下载的 **products.csv** 文件上传到计算机。
1. 在“**创建或修改文件上传中的表格**”页中，确保选择页面右上角的群集。 然后选择 **hive_metastore** 目录及其默认架构以新建名为“**products**”的表格。
1. 创建**产品**表后，在“**目录资源管理器**”页中的“**创建**”按钮菜单中，选择“**笔记本**”以创建笔记本。
1. 在笔记本中，确保笔记本已连接到群集，然后查看已自动添加到第一个单元格的代码，应如下所示：

    ```python
    %sql
    SELECT * FROM `hive_metastore`.`default`.`products`;
    ```

1. 使用单元格左侧的“**&#9656; 运行单元格**”菜单选项来运行该代码，启动并在出现提示时附加群集。
1. 等待代码运行的 Spark 作业完成。 该代码从基于上传文件创建的表格中检索数据。
1. 在结果表上方，选择 +，然后选择“可视化效果”以查看可视化效果编辑器，然后应用以下选项********：
    - **可视化效果类型**：条形图
    - **X 列**：类别
    - **Y 列**：添加新列并选择“ProductID”******。 应用“计数”聚合********。

    保存可视化效果，并观察它是否显示在笔记本中，如下所示：

    ![按类别显示产品计数的条形图](./images/databricks-chart.png)

## 使用数据帧分析数据

虽然大多数数据分析师都能熟练使用上一示例中使用的 SQL 代码，但部分数据分析师和数据科学家可以在 *PySpark*（Python 的 Spark 优化版本）等编程语言中使用本机 Spark 对象（如*数据帧*）来高效处理数据。

1. 在笔记本中，在先前运行的代码单元格的图表输出下，使用 **+ 代码**图标添加新单元格。

    > **提示**：可能需要在输出单元格下移动鼠标，以使 **+ 代码**图标出现。

1. 在新单元格中，输入并运行以下代码：

    ```python
    df = spark.sql("SELECT * FROM products")
    df = df.filter("Category == 'Road Bikes'")
    display(df)
    ```

1. 运行新单元格，该单元格返回“*公路自行车*”类别中的产品。

## 清理

在 Azure Databricks 门户的“**计算**”页上，选择群集，然后选择“**&#9632; 终止**”以将其关闭。

如果已完成对 Azure Databricks 的探索，则可以删除已创建的资源，以避免产生不必要的 Azure 成本并释放订阅中的容量。
