---
lab:
  title: 通过 Azure Databricks 使用 Microsoft Purview 和 Unity Catalog 实现数据隐私和治理
---

# 通过 Azure Databricks 使用 Microsoft Purview 和 Unity Catalog 实现数据隐私和治理

Microsoft Purview 允许跨整个数据资产进行全面的数据治理，可与 Azure Databricks 完美集成以管理 Lakehouse 数据并将元数据引入数据映射。 Unity Catalog 提供集中式数据管理和治理，简化了 Databricks 工作区中的安全性和合规性，从而对这一点进行了提升。

完成本实验室大约需要 30 分钟。

## 预配 Azure Databricks 工作区

> **提示**：如果你已有 Azure Databricks 工作区，则可以跳过此过程并使用现有工作区。

本练习包括一个用于预配新 Azure Databricks 工作区的脚本。 该脚本会尝试在一个区域中创建*高级*层 Azure Databricks 工作区资源，在该区域中，Azure 订阅具有本练习所需计算核心的充足配额；该脚本假设你的用户帐户在订阅中具有足够的权限来创建 Azure Databricks 工作区资源。 如果脚本由于配额或权限不足失败，可以尝试 [在 Azure 门户中以交互方式创建 Azure Databricks 工作区](https://learn.microsoft.com/azure/databricks/getting-started/#--create-an-azure-databricks-workspace)。

1. 在 Web 浏览器中，登录到 [Azure 门户](https://portal.azure.com)，网址为 `https://portal.azure.com`。

2. 使用页面顶部搜索栏右侧的 [\>_] 按钮在 Azure 门户中创建新的 Cloud Shell，在出现提示时选择“PowerShell”环境并创建存储。 Cloud Shell 在 Azure 门户底部的窗格中提供命令行界面，如下所示：

    ![具有 Cloud Shell 窗格的 Azure 门户](./images/cloud-shell.png)

    > 注意：如果以前创建了使用 Bash 环境的 Cloud shell，请使用 Cloud Shell 窗格左上角的下拉菜单将其更改为“PowerShell”。

3. 请注意，可以通过拖动窗格顶部的分隔条或使用窗格右上角的 &#8212;、&#9723; 或 X 图标来调整 Cloud Shell 的大小，以最小化、最大化和关闭窗格  。 有关如何使用 Azure Cloud Shell 的详细信息，请参阅 [Azure Cloud Shell 文档](https://docs.microsoft.com/azure/cloud-shell/overview)。

4. 在 PowerShell 窗格中，输入以下命令以克隆此存储库：

     ```powershell
    rm -r mslearn-databricks -f
    git clone https://github.com/MicrosoftLearning/mslearn-databricks
     ```

5. 克隆存储库后，请输入以下命令以运行 **setup.ps1** 脚本，以在可用区域中预配 Azure Databricks 工作区：

     ```powershell
    ./mslearn-databricks/setup.ps1
     ```

6. 如果出现提示，请选择要使用的订阅（仅当有权访问多个 Azure 订阅时才会发生这种情况）。

7. 等待脚本完成 - 这通常需要大约 5 分钟，但在某些情况下可能需要更长的时间。 在等待时，请查看 Azure Databricks 文档中的 [Delta Lake 简介](https://docs.microsoft.com/azure/databricks/delta/delta-intro)一文。

## 创建群集

Azure Databricks 是一个分布式处理平台，可使用 Apache Spark 群集在多个节点上并行处理数据。 每个群集由一个用于协调工作的驱动程序节点和多个用于执行处理任务的工作器节点组成。 在本练习中，将创建一个*单节点*群集，以最大程度地减少实验室环境中使用的计算资源（在实验室环境中，资源可能会受到限制）。 在生产环境中，通常会创建具有多个工作器节点的群集。

> **提示**：如果 Azure Databricks 工作区中已有一个具有 13.3 LTS ML 或更高运行时版本的群集，则可以使用它来完成此练习并跳过此过程。

1. 在 Azure 门户中，浏览到已由脚本创建的 **msl-xxxxxxx*** 资源组（或包含现有 Azure Databricks 工作区的资源组）

1. 选择 Azure Databricks 服务资源（如果已使用安装脚本创建，则名为 **databricks-xxxxxxx***）。

1. 在工作区的“概述”**** 页中，使用“启动工作区”**** 按钮在新的浏览器标签页中打开 Azure Databricks 工作区；请在出现提示时登录。

    > 提示：使用 Databricks 工作区门户时，可能会显示各种提示和通知。 消除这些内容，并按照提供的说明完成本练习中的任务。

1. 在左侧边栏中，选择“**(+) 新建**”任务，然后选择“**群集**”。

1. 在“新建群集”页中，使用以下设置创建新群集：
    - 群集名称：用户名的群集（默认群集名称）
    - **策略**：非受限
    - 群集模式：单节点
    - 访问模式：单用户（选择你的用户帐户）
    - **Databricks 运行时版本**：13.3 LTS（Spark 3.4.1、Scala 2.12）或更高版本
    - 使用 Photon 加速：已选择
    - 节点类型：Standard_DS3_v2
    - 在处于不活动状态 20 分钟后终止**********

1. 等待群集创建完成。 这可能需要一到两分钟时间。

    > 注意：如果群集无法启动，则订阅在预配 Azure Databricks 工作区的区域中的配额可能不足。 请参阅 [CPU 内核限制阻止创建群集](https://docs.microsoft.com/azure/databricks/kb/clusters/azure-core-limit)，了解详细信息。 如果发生这种情况，可以尝试删除工作区，并在其他区域创建新工作区。 可以将区域指定为设置脚本的参数，如下所示：`./mslearn-databricks/setup.ps1 eastus`

## 设置 Unity Catalog

Unity Catalog 元存储会注册有关安全对象（例如表、卷、外部位置和共享）及其访问管理权限的元存储。 每个元存储公开一个三级命名空间 (`catalog`.`schema`.`table`)，可在该命名空间组织数据。 你必须对组织在其中操作的每个区域具有一个元存储。 若要使用 Unity Catalog，用户必须位于附加到其区域中元存储的工作区上。

1. 在边栏中，选择“目录”****。

2. 在目录资源管理器中，应存在包含工作区名称的默认 Unity 目录（如果已使用设置脚本创建它，则为 databricks-*xxxxxxx*****）。 选择目录，然后在右窗格顶部选择“创建架构”****。

3. 将新架构命名为电子商务****，选择使用工作区创建的存储位置，然后选择“创建”****。

4. 选择目录，然后在右窗格中选择“工作区”**** 选项卡。验证工作区对其是否具有 `Read & Write` 访问权限。

## 将示例数据引入 Azure Databricks

1. 下载示例数据文件：
   * [customers.csv](https://github.com/MicrosoftLearning/mslearn-databricks/raw/main/data/DE-05/customers.csv)
   * [Products.csv](https://github.com/MicrosoftLearning/mslearn-databricks/raw/main/data/DE-05/products.csv)
   * [sales.csv](https://github.com/MicrosoftLearning/mslearn-databricks/raw/main/data/DE-05/sales.csv)

2. 在 Azure Databricks 工作区的目录资源管理器顶部，选择+**** 并选择“添加数据”****。

3. 在新窗口中，选择“将文件上传到卷”****。

4. 在新窗口中，导航到 `ecommerce` 架构，展开它并选择“创建卷”****。

5. 将新卷命名为 sample_data****，然后选择“创建”****。

6. 选择新卷并上传文件 `customers.csv`、`products.csv` 以及 `sales.csv`。 选择“上传”。

7. 在边栏中，使用“(+) 新建”**** 链接创建**笔记本**。 在“连接”**** 下拉列表中，选择群集（如果尚未选择）。 如果群集未运行，可能需要一分钟左右才能启动。

8. 在笔记本的第一个单元格中，输入以下代码以从 CSV 文件创建表：

     ```python
    # Load Customer Data
    customers_df = spark.read.format("csv").option("header", "true").load("/Volumes/databricksxxxxxxx/ecommerce/sample_data/customers.csv")
    customers_df.write.saveAsTable("ecommerce.customers")

    # Load Sales Data
    sales_df = spark.read.format("csv").option("header", "true").load("/Volumes/databricksxxxxxxx/ecommerce/sample_data/sales.csv")
    sales_df.write.saveAsTable("ecommerce.sales")

    # Load Product Data
    products_df = spark.read.format("csv").option("header", "true").load("/Volumes/databricksxxxxxxx/ecommerce/sample_data/products.csv")
    products_df.write.saveAsTable("ecommerce.products")
     ```

>**备注：** 在 `.load` 文件路径中，将 `databricksxxxxxxx` 替换为目录名称。

9. 在目录资源管理器中，导航到 `sample_data` 卷并验证新表是否位于其中。
    
## 设置 Microsoft Purview

Microsoft Purview 是一种统一的数据治理服务，可帮助组织在各种环境中管理数据并保护其安全。 借助数据丢失防护、信息保护和合规性管理等功能，Microsoft Purview 提供在整个生命周期内了解、管理和保护数据的工具。

1. 导航到 [Azure 门户](https://portal.azure.com/)。

2. 选择“创建资源”**** 并搜索“Microsoft Purview”****。

3. 使用以下设置创建 Microsoft Purview**** 资源：
    - 订阅****：*选择 Azure 订阅*
    - **资源组**：*选择与 Azure Databricks 工作区相同的资源组*
    - **Microsoft Purview 帐户名称**：*所选的唯一名称*
    - **位置**：*选择与 Azure Databricks 工作区相同的区域*

4. 选择“查看 + 创建”  。 等待验证完成，然后选择“创建”****。

5. 等待部署完成。 然后，在 Azure 门户中转到已部署的 Microsoft Purview 资源。

6. 在 Microsoft Purview 治理门户中，导航到边栏中的“数据映射”**** 部分。

7. 在“数据源”**** 窗格中，选择“注册”****。

8. 在“注册数据源”**** 窗口中，搜索并选择“Azure Databricks”****。 选择**继续**。

9. 为数据源指定唯一名称，然后选择 Azure Databricks 工作区。 选择**注册**。

## 实现数据隐私和治理策略

1. 在边栏的“数据映射”**** 部分中，选择“分类”****。

2. 在“分类”**** 窗格中，选择“+ 新建”**** 并新建名为 PII**** 的分类（个人身份信息）。 选择“确定”****。

3. 在边栏中选择“数据目录”**** 并导航到“客户”**** 表。

4. 将 PII 分类应用于电子邮件和电话列。

5. 转到 Azure Databricks 并打开以前创建的笔记本。
 
6. 在新建单元格中，运行以下代码以创建数据访问策略，以限制对 PII 数据的访问。

     ```sql
    CREATE OR REPLACE TABLE ecommerce.customers (
      customer_id STRING,
      name STRING,
      email STRING,
      phone STRING,
      address STRING,
      city STRING,
      state STRING,
      zip_code STRING,
      country STRING
    ) TBLPROPERTIES ('data_classification'='PII');

    GRANT SELECT ON TABLE ecommerce.customers TO ROLE data_scientist;
    REVOKE SELECT (email, phone) ON TABLE ecommerce.customers FROM ROLE data_scientist;
     ```

7. 尝试将客户表查询为具有 data_scientist 角色的用户。 验证对 PII 列（电子邮件和电话）的访问是否受到限制。

## 清理

在 Azure Databricks 门户的“**计算**”页上，选择群集，然后选择“**&#9632; 终止**”以将其关闭。

如果已完成对 Azure Databricks 的探索，则可以删除已创建的资源，以避免产生不必要的 Azure 成本并释放订阅中的容量。
