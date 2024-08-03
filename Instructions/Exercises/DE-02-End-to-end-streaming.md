---
lab:
  title: Azure Databricks 中使用增量实时表的端到端流式处理管道
---

# Azure Databricks 中使用增量实时表的端到端流式处理管道

使用 Azure Databricks 中的“增量实时表”创建端到端流式处理管道涉及定义数据转换，然后增量实时表通过任务业务流程、群集管理和监视进行管理。 此框架支持流式处理表来处理持续更新的数据、复杂转换的具体化视图，以及中间转换和数据质量检查的视图。

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

## 创建笔记本并引入数据

1. 在边栏中，使用“(+) 新建”**** 链接创建**笔记本**。 在“连接”**** 下拉列表中，选择群集（如果尚未选择）。 如果群集未运行，可能需要一分钟左右才能启动。

3. 在笔记本的第一个单元格中输入以下代码，该代码使用 *shell* 命令将数据文件从 GitHub 下载到群集使用的文件系统中。

     ```python
    %sh
    rm -r /dbfs/device_stream
    mkdir /dbfs/device_stream
    wget -O /dbfs/device_stream/device_data.csv https://github.com/MicrosoftLearning/mslearn-databricks/raw/main/data/device_data.csv
     ```

4. 使用单元格左侧的“&#9656; 运行单元格”菜单选项来运行该代码****。 然后等待代码运行的 Spark 作业完成。

## 使用 Delta 表对数据进行流式处理

Delta Lake 支持流式处理数据。** Delta 表可以是接收器，也可以是使用 Spark 结构化流式处理 API 创建的数据流的数据源 。 在此示例中，你将使用 Delta 表作为模拟物联网 (IoT) 方案中部分流式处理数据的接收器。 在下一个任务中，此增量表将充当数据转换的实时源。

1. 在新单元格中，运行以下代码，基于包含 CSV 设备数据的文件夹创建流：

     ```python
    from pyspark.sql.functions import *
    from pyspark.sql.types import *

    # Define the schema for the incoming data
    schema = StructType([
        StructField("device_id", StringType(), True),
        StructField("timestamp", TimestampType(), True),
        StructField("temperature", DoubleType(), True),
        StructField("humidity", DoubleType(), True)
    ])

    # Read streaming data from folder
    inputPath = '/device_stream/'
    iotstream = spark.readStream.schema(schema).option("header", "true").csv(inputPath)
    print("Source stream created...")

    # Write the data to a Delta table
    query = (iotstream
             .writeStream
             .format("delta")
             .option("checkpointLocation", "/tmp/checkpoints/iot_data")
             .start("/tmp/delta/iot_data"))
     ```

2. 使用单元格左侧的“&#9656; 运行单元格”菜单选项来运行该代码****。

此增量表现在将成为数据转换的实时源。

   > 注意：上方代码单元会创建源流。 因此，作业运行永远不会更改为已完成状态。 要手动停止流式处理，可以在新单元格中运行 `query.stop()`。
   
## 创建增量实时表管道

管道是使用“增量实时表”配置和运行数据处理工作流的主单元。 通过 Python 或 SQL 中的有向无环图 (DAG) 将数据源链接到目标数据集。

1. 在左侧边栏中选择“增量实时表”**** ，然后选择“创建管道”****。

2. 在“创建管道”**** 页中，使用以下设置创建新管道：
    - **管道名称**：设置管道的名称
    - **产品版本**：高级
    - **管道模式**：触发
    - **源代码**：将其留空
    - **存储选项**：Hive 元存储
    - **存储位置**：dbfs:/pipelines/device_stream

3. 选择**创建**。

4. 创建管道后，在右侧面板中的“源代码”**** 下，打开指向空白笔记本的链接：

    ![delta-live-table-pipeline](./images/delta-live-table-pipeline.png)

5. 在笔记本的第一个单元中，输入以下代码以创建“增量实时表”并转换数据：

     ```python
    import dlt
    from pyspark.sql.functions import col, current_timestamp
     
    @dlt.table(
        name="raw_iot_data",
        comment="Raw IoT device data"
    )
    def raw_iot_data():
        return spark.readStream.format("delta").load("/tmp/delta/iot_data")

    @dlt.table(
        name="transformed_iot_data",
        comment="Transformed IoT device data with derived metrics"
    )
    def transformed_iot_data():
        return (
            dlt.read("raw_iot_data")
            .withColumn("temperature_fahrenheit", col("temperature") * 9/5 + 32)
            .withColumn("humidity_percentage", col("humidity") * 100)
            .withColumn("event_time", current_timestamp())
        )
     ```

6. 选择**开始**。

7. 成功执行管道后，返回到第一个笔记本，并使用以下代码验证是否已在指定的存储位置创建新表：

     ```sql
    display(dbutils.fs.ls("dbfs:/pipelines/device_stream/tables"))
     ```

## 以可视化效果的形式查看结果

创建表后，可以将这些表加载到数据帧并可视化数据。

1. 在第一个笔记本中，添加新的代码单元并运行以下代码，将 `transformed_iot_data` 加载到数据帧中：

    ```python
   df = spark.read.format("delta").load('/pipelines/device_stream/tables/transformed_iot_data')
   display(df)
    ```

1. 在结果表上方，选择 +，然后选择“可视化效果”以查看可视化效果编辑器，然后应用以下选项********：
    - **可视化类型**：折线图
    - **X 列**：时间戳
    - **Y 列**：*添加新列并选择*“temperature_fahrenheit”****。 *应用* **Sum** *聚合*。

1. 保存可视化效果，然后查看笔记本中生成的图表。

## 清理

在 Azure Databricks 门户的“**计算**”页上，选择群集，然后选择“**&#9632; 终止**”以将其关闭。

如果已完成对 Azure Databricks 的探索，则可以删除已创建的资源，以避免产生不必要的 Azure 成本并释放订阅中的容量。
