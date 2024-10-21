---
lab:
  title: 已弃用 - 在 Azure Databricks 中使用 Delta Lake
---

# 在 Azure Databricks 中使用 Delta Lake

Delta Lake 是一个开源项目，用于在数据湖之上为 Spark 生成事务数据存储层。 Delta Lake 为批处理和流式处理数据操作添加了对关系语义的支持，并支持创建 Lakehouse 体系结构。在该体系结构中，Apache Spark 可用于处理和查询基于数据湖中基础文件的表中的数据。

完成本实验室大约需要 40 分钟。

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

现在，让我们创建一个 Spark 笔记本并导入将在本练习中使用的数据。

1. 在边栏中，使用“(+) 新建”**** 链接创建**笔记本**。
1. 将默认笔记本名称（**Untitled Notebook *[日期]***）更改为“Explore Delta Lake”，然后在“连接”下拉列表中选择群集（如果尚未选中）****。**** 如果群集未运行，可能需要一分钟左右才能启动。
1. 在笔记本的第一个单元格中输入以下代码，该代码使用 *shell* 命令将数据文件从 GitHub 下载到群集使用的文件系统中。

    ```python
    %sh
    rm -r /dbfs/delta_lab
    mkdir /dbfs/delta_lab
    wget -O /dbfs/delta_lab/products.csv https://raw.githubusercontent.com/MicrosoftLearning/mslearn-databricks/main/data/products.csv
    ```

1. 使用单元格左侧的“&#9656; 运行单元格”菜单选项来运行该代码****。 然后等待代码运行的 Spark 作业完成。
1. 在现有代码单元格下，使用 + 图标添加新的代码单元格****。 然后在新单元格中输入并运行以下代码，以便从文件加载数据并查看前 10 行。

    ```python
   df = spark.read.load('/delta_lab/products.csv', format='csv', header=True)
   display(df.limit(10))
    ```

## 将文件数据加载到 delta 表中

数据已加载到数据帧中。 让我们将其持久保存到 Delta 表中。

1. 添加一个新代码单元格并使用它来运行以下代码：

    ```python
   delta_table_path = "/delta/products-delta"
   df.write.format("delta").save(delta_table_path)
    ```

    Delta Lake 表的数据以 Parquet 格式存储。 此外还创建一个日志文件来跟踪对数据所做的修改。

1. 添加一个新的代码单元格并使用它运行以下 shell 命令来查看保存 Delta 数据的文件夹的内容。

    ```
    %sh
    ls /dbfs/delta/products-delta
    ```

1. Delta 格式的文件数据可以加载到 DeltaTable**** 对象中，你可以使用该对象查看和更新表中的数据。 在新单元格中运行以下代码来更新数据；将产品 771 的价格降低 10%。

    ```python
   from delta.tables import *
   from pyspark.sql.functions import *
   
   # Create a deltaTable object
   deltaTable = DeltaTable.forPath(spark, delta_table_path)
   # Update the table (reduce price of product 771 by 10%)
   deltaTable.update(
       condition = "ProductID == 771",
       set = { "ListPrice": "ListPrice * 0.9" })
   # View the updated data as a dataframe
   deltaTable.toDF().show(10)
    ```

    更新会持久保留到 Delta 文件夹中的数据中，并且会反映在从该位置加载的任何新数据帧中。

1. 运行以下代码，以便从 Delta 表数据创建一个新的数据帧：

    ```python
   new_df = spark.read.format("delta").load(delta_table_path)
   new_df.show(10)
    ```

## 探索日志记录和“按时间顺序查看”**

数据修改会被记录下来，使你能够使用 Delta Lake 的“按时间顺序查看”** 功能来查看数据的先前版本。 

1. 在新建代码单元格中，使用以下代码查看产品数据的原始版本：

    ```python
   new_df = spark.read.format("delta").option("versionAsOf", 0).load(delta_table_path)
   new_df.show(10)
    ```

1. 该日志包含数据修改的完整历史记录。 使用以下代码查看最近 10 次更改的记录：

    ```python
   deltaTable.history(10).show(10, False, True)
    ```

## 创建目录表

到目前为止，你已通过从包含表所基于 parquet 文件的文件夹加载数据来处理 Delta 表。 可以定义封装数据的目录表，并提供可以在 SQL 代码中引用的命名表实体**。 对于 Delta Lake，Spark 支持两种类型的目录表：

- 由指向包含表数据的文件的路径定义的 *外部* 表。
- 在元存储中定义的 *托管表*。

### 创建外部表

1. 使用以下代码来创建名为“AdventureWorks”的新数据库，然后基于前面定义的 Delta 文件的路径在该数据库中创建名为“ProductsExternal”的外部表********：

    ```python
   spark.sql("CREATE DATABASE AdventureWorks")
   spark.sql("CREATE TABLE AdventureWorks.ProductsExternal USING DELTA LOCATION '{0}'".format(delta_table_path))
   spark.sql("DESCRIBE EXTENDED AdventureWorks.ProductsExternal").show(truncate=False)
    ```

    请注意，新表的 Location 属性是指定的路径****。

1. 使用以下代码查询该表：

    ```sql
   %sql
   USE AdventureWorks;
   SELECT * FROM ProductsExternal;
    ```

### 创建托管表

1. 运行以下代码，根据你最初从 products.csv**** 文件加载的数据帧（在更新产品 771 的价格之前）创建（并描述）名为 ProductsManaged**** 的托管表。

    ```python
   df.write.format("delta").saveAsTable("AdventureWorks.ProductsManaged")
   spark.sql("DESCRIBE EXTENDED AdventureWorks.ProductsManaged").show(truncate=False)
    ```

    你没有为表使用的 parquet 文件指定路径 - 在 Hive 元存储中系统会为你管理此项，并将其显示在表说明的 **位置** 属性中。

1. 使用以下代码来查询托管表，请注意，该语法与托管表的语法相同：

    ```sql
   %sql
   USE AdventureWorks;
   SELECT * FROM ProductsManaged;
    ```

### 比较外部表和托管表

1. 使用以下代码列出 AdventureWorks**** 数据库中的表：

    ```sql
   %sql
   USE AdventureWorks;
   SHOW TABLES;
    ```

1. 现在使用以下代码查看这些表所基于的文件夹：

    ```Bash
    %sh
    echo "External table:"
    ls /dbfs/delta/products-delta
    echo
    echo "Managed table:"
    ls /dbfs/user/hive/warehouse/adventureworks.db/productsmanaged
    ```

1. 使用以下代码从数据库删除这两个表：

    ```sql
   %sql
   USE AdventureWorks;
   DROP TABLE IF EXISTS ProductsExternal;
   DROP TABLE IF EXISTS ProductsManaged;
   SHOW TABLES;
    ```

1. 现在重新运行包含以下代码的单元格，以查看 Delta 文件夹的内容：

    ```Bash
    %sh
    echo "External table:"
    ls /dbfs/delta/products-delta
    echo
    echo "Managed table:"
    ls /dbfs/user/hive/warehouse/adventureworks.db/productsmanaged
    ```

    当托管表被删除时，其文件会被自动删除。 但是，外部表的文件仍然存在。 删除外部表只会从数据库中删除表元数据，不会删除数据文件。

1. 使用以下代码在数据库中创建一个基于 products-delta**** 文件夹中的 Delta 文件的新表：

    ```sql
   %sql
   USE AdventureWorks;
   CREATE TABLE Products
   USING DELTA
   LOCATION '/delta/products-delta';
    ```

1. 使用以下代码查询该新表：

    ```sql
   %sql
   USE AdventureWorks;
   SELECT * FROM Products;
    ```

    由于该表基于现有的 Delta 文件（其中包括已记录的更改历史），因此它反映了之前对产品数据所做的修改。

## 使用 Delta 表对数据进行流式处理

Delta Lake 支持流式处理数据。** Delta 表可以是接收器，也可以是使用 Spark 结构化流式处理 API 创建的数据流的数据源 。 在此示例中，你将使用 Delta 表作为模拟物联网 (IoT) 方案中部分流式处理数据的接收器。 模拟的设备数据是 JSON 格式的，如下所示：

```json
{"device":"Dev1","status":"ok"}
{"device":"Dev1","status":"ok"}
{"device":"Dev1","status":"ok"}
{"device":"Dev2","status":"error"}
{"device":"Dev1","status":"ok"}
{"device":"Dev1","status":"error"}
{"device":"Dev2","status":"ok"}
{"device":"Dev2","status":"error"}
{"device":"Dev1","status":"ok"}
```

1. 在新单元格中，运行以下代码以下载 JSON 文件：

    ```bash
    %sh
    rm -r /dbfs/device_stream
    mkdir /dbfs/device_stream
    wget -O /dbfs/device_stream/devices1.json https://raw.githubusercontent.com/MicrosoftLearning/mslearn-databricks/main/data/devices1.json
    ```

1. 在新单元格中，运行以下代码以基于包含 JSON 设备数据的文件夹创建流：

    ```python
   from pyspark.sql.types import *
   from pyspark.sql.functions import *
   
   # Create a stream that reads data from the folder, using a JSON schema
   inputPath = '/device_stream/'
   jsonSchema = StructType([
   StructField("device", StringType(), False),
   StructField("status", StringType(), False)
   ])
   iotstream = spark.readStream.schema(jsonSchema).option("maxFilesPerTrigger", 1).json(inputPath)
   print("Source stream created...")
    ```

1. 添加一个新的代码单元格并使用它将数据流永久写入 Delta 文件夹：

    ```python
   # Write the stream to a delta table
   delta_stream_table_path = '/delta/iotdevicedata'
   checkpointpath = '/delta/checkpoint'
   deltastream = iotstream.writeStream.format("delta").option("checkpointLocation", checkpointpath).start(delta_stream_table_path)
   print("Streaming to delta sink...")
    ```

1. 添加代码来读取数据，就像任何其他 Delta 文件夹一样：

    ```python
   # Read the data in delta format into a dataframe
   df = spark.read.format("delta").load(delta_stream_table_path)
   display(df)
    ```

1. 添加以下代码，以基于要将流数据写入其中的 Delta 文件夹创建一个表：

    ```python
   # create a catalog table based on the streaming sink
   spark.sql("CREATE TABLE IotDeviceData USING DELTA LOCATION '{0}'".format(delta_stream_table_path))
    ```

1. 使用以下代码查询该表：

    ```sql
   %sql
   SELECT * FROM IotDeviceData;
    ```

1. 运行以下代码，将一些新的设备数据添加到流中：

    ```Bash
    %sh
    wget -O /dbfs/device_stream/devices2.json https://raw.githubusercontent.com/MicrosoftLearning/mslearn-databricks/main/data/devices2.json
    ```

1. 重新运行以下 SQL 查询代码，以验证新数据是否已添加到流中并写入 Delta 文件夹：

    ```sql
   %sql
   SELECT * FROM IotDeviceData;
    ```

1. 运行以下代码来停止流：

    ```python
   deltastream.stop()
    ```

## 清理

在 Azure Databricks 门户的“**计算**”页上，选择群集，然后选择“**&#9632; 终止**”以将其关闭。

如果已完成对 Azure Databricks 的探索，则可以删除已创建的资源，以避免产生不必要的 Azure 成本并释放订阅中的容量。