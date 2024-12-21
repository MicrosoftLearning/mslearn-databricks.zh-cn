---
lab:
  title: 使用 Azure Databricks 管理机器学习模型
---

# 使用 Azure Databricks 管理机器学习模型

使用 Azure Databricks 训练机器学习模型涉及利用统一分析平台，该平台为数据处理、模型训练和部署提供协作环境。 Azure Databricks 与 MLflow 集成，用于管理机器学习生命周期，包括试验跟踪和模型服务。

完成此练习大约需要 **20** 分钟。

## 开始之前

需要一个你在其中具有管理级权限的 [Azure 订阅](https://azure.microsoft.com/free)。

## 预配 Azure Databricks 工作区

> **提示**：如果你已有 Azure Databricks 工作区，则可以跳过此过程并使用现有工作区。

本练习包括一个用于预配新 Azure Databricks 工作区的脚本。 该脚本会尝试在一个区域中创建*高级*层 Azure Databricks 工作区资源，在该区域中，Azure 订阅具有本练习所需计算核心的充足配额；该脚本假设你的用户帐户在订阅中具有足够的权限来创建 Azure Databricks 工作区资源。 如果脚本由于配额或权限不足失败，可以尝试 [在 Azure 门户中以交互方式创建 Azure Databricks 工作区](https://learn.microsoft.com/azure/databricks/getting-started/#--create-an-azure-databricks-workspace)。

1. 在 Web 浏览器中，登录到 [Azure 门户](https://portal.azure.com)，网址为 `https://portal.azure.com`。
2. 使用页面顶部搜索栏右侧的 **[\>_]** 按钮在 Azure 门户中创建新的 Cloud Shell，选择 ***PowerShell*** 环境。 Cloud Shell 在 Azure 门户底部的窗格中提供命令行界面，如下所示：

    ![具有 Cloud Shell 窗格的 Azure 门户](./images/cloud-shell.png)

    > **备注**：如果以前创建了使用 *Bash* 环境的 Cloud Shell，请将其切换到 ***PowerShell***。

3. 请注意，可以通过拖动窗格顶部的分隔条来调整 Cloud Shell 的大小或使用窗格右上角的 **&#8212;**、**&#10530;** 或 **X** 图标来最小化、最大化和关闭窗格。 有关如何使用 Azure Cloud Shell 的详细信息，请参阅 [Azure Cloud Shell 文档](https://docs.microsoft.com/azure/cloud-shell/overview)。

4. 在 PowerShell 窗格中，输入以下命令以克隆此存储库：

    ```
    rm -r mslearn-databricks -f
    git clone https://github.com/MicrosoftLearning/mslearn-databricks
    ```

5. 克隆存储库后，请输入以下命令以运行 **setup.ps1** 脚本，以在可用区域中预配 Azure Databricks 工作区：

    ```
    ./mslearn-databricks/setup.ps1
    ```

6. 如果出现提示，请选择要使用的订阅（仅当有权访问多个 Azure 订阅时才会发生这种情况）。
7. 等待脚本完成 - 这通常需要大约 5 分钟，但在某些情况下可能需要更长的时间。 在等待时，请查看 Azure Databricks 文档中的[什么是 Databricks 机器学习？](https://learn.microsoft.com/azure/databricks/machine-learning/)一文。

## 创建群集

Azure Databricks 是一个分布式处理平台，可使用 Apache Spark 群集在多个节点上并行处理数据。 每个群集由一个用于协调工作的驱动程序节点和多个用于执行处理任务的工作器节点组成。 在本练习中，将创建一个*单节点*群集，以最大程度地减少实验室环境中使用的计算资源（在实验室环境中，资源可能会受到限制）。 在生产环境中，通常会创建具有多个工作器节点的群集。

> **提示**：如果 Azure Databricks 工作区中已有一个具有 13.3 LTS ML**<u></u>** 或更高运行时版本的群集，则可以使用它来完成此练习并跳过此过程。

1. 在 Azure 门户中，浏览到已由脚本创建的 msl-xxxxxxx****** 资源组（或包含现有 Azure Databricks 工作区的资源组）
1. 选择 Azure Databricks 服务资源（如果已使用安装脚本创建，则名为 **databricks-xxxxxxx***）。
1. 在工作区的“概述”**** 页中，使用“启动工作区”**** 按钮在新的浏览器标签页中打开 Azure Databricks 工作区；请在出现提示时登录。

    > 提示：使用 Databricks 工作区门户时，可能会显示各种提示和通知。 消除这些内容，并按照提供的说明完成本练习中的任务。

1. 在左侧边栏中，选择“**(+) 新建**”任务，然后选择“**群集**”。
1. 在“新建群集”页中，使用以下设置创建新群集：
    - 群集名称：用户名的群集（默认群集名称）
    - **策略**：非受限
    - 群集模式：单节点
    - 访问模式：单用户（选择你的用户帐户）
    - Databricks Runtime 版本****：选择最新非 beta 版本运行时的 ML***<u></u>** 版本（不是****标准运行时版本），该版本符合以下条件：*
        - 不使用 GPU**
        - 包括 Scala > 2.11
        - *包括 Spark > **3.4***
    - 使用 Photon 加速****：未选定<u></u>
    - **节点类型**：Standard_D4ds_v5
    - 在处于不活动状态 20 分钟后终止**********

1. 等待群集创建完成。 这可能需要一到两分钟时间。

> 注意：如果群集无法启动，则订阅在预配 Azure Databricks 工作区的区域中的配额可能不足。 请参阅 [CPU 内核限制阻止创建群集](https://docs.microsoft.com/azure/databricks/kb/clusters/azure-core-limit)，了解详细信息。 如果发生这种情况，可以尝试删除工作区，并在其他区域创建新工作区。 可以将区域指定为设置脚本的参数，如下所示：`./mslearn-databricks/setup.ps1 eastus`

## 创建笔记本

你将运行使用 Spark MLLib 库来训练机器学习模型的代码，因此第一步是在工作区中创建一个新笔记本。

1. 在边栏中，使用“(+) 新建”**** 链接创建**笔记本**。
1. 将默认笔记本名称（**Untitled Notebook *[日期]***）更改为“机器学习”，然后在“连接”下拉列表中选择群集（如果尚未选中）****。**** 如果群集未运行，可能需要一分钟左右才能启动。

## 引入和准备数据

本练习的场景基于对南极洲企鹅的观察，目的是训练一个机器学习模型，用于根据观察到的企鹅的位置和身体度量来预测其种类。

> **引文**：本练习中使用的企鹅数据集是 [Kristen Gorman 博 士](https://www.uaf.edu/cfos/people/faculty/detail/kristen-gorman.php)和[长期生态研究网络](https://lternet.edu/)成员[南极洲帕默站](https://pal.lternet.edu/)收集并提供的数据的子集。

1. 在笔记本的第一个单元格中输入以下代码，该代码使用 shell** 命令将企鹅数据从 GitHub 下载到群集使用的文件系统中。

    ```bash
    %sh
    rm -r /dbfs/ml_lab
    mkdir /dbfs/ml_lab
    wget -O /dbfs/ml_lab/penguins.csv https://raw.githubusercontent.com/MicrosoftLearning/mslearn-databricks/main/data/penguins.csv
    ```

1. 使用单元格左侧的“&#9656; 运行单元格”菜单选项来运行该代码****。 然后等待代码运行的 Spark 作业完成。

1. 现在为机器学习准备数据。 在现有代码单元格下，使用 + 图标添加新的代码单元格****。 然后在新单元格中输入并运行以下代码，其用途为：
    - 删除任何不完整的行
    - 应用适当的数据类型
    - 查看数据的随机样本
    - 将数据拆分成两个数据集：一个用于训练，另一个用于测试。

    ```python
   from pyspark.sql.types import *
   from pyspark.sql.functions import *
   
   data = spark.read.format("csv").option("header", "true").load("/hyperopt_lab/penguins.csv")
   data = data.dropna().select(col("Island").astype("string"),
                             col("CulmenLength").astype("float"),
                             col("CulmenDepth").astype("float"),
                             col("FlipperLength").astype("float"),
                             col("BodyMass").astype("float"),
                             col("Species").astype("int")
                             )
   display(data.sample(0.2))
   
   splits = data.randomSplit([0.7, 0.3])
   train = splits[0]
   test = splits[1]
   print ("Training Rows:", train.count(), " Testing Rows:", test.count())
    ```
    
## 运行管道来预处理数据并训练 ML 模型

训练模型之前，需执行特征工程步骤，然后将算法拟合到数据。 若要将模型与某些测试数据一起使用来生成预测，则必须对测试数据应用相同的特征工程步骤。 一种生成和使用模型的更高效方法是封装用于准备数据的转换器，以及用于在*管道*中训练它的模型。

1. 使用以下代码创建封装数据准备和模型训练步骤的管道：

    ```python
   from pyspark.ml import Pipeline
   from pyspark.ml.feature import StringIndexer, VectorAssembler, MinMaxScaler
   from pyspark.ml.classification import LogisticRegression
   
   catFeature = "Island"
   numFeatures = ["CulmenLength", "CulmenDepth", "FlipperLength", "BodyMass"]
   
   # Define the feature engineering and model training algorithm steps
   catIndexer = StringIndexer(inputCol=catFeature, outputCol=catFeature + "Idx")
   numVector = VectorAssembler(inputCols=numFeatures, outputCol="numericFeatures")
   numScaler = MinMaxScaler(inputCol = numVector.getOutputCol(), outputCol="normalizedFeatures")
   featureVector = VectorAssembler(inputCols=["IslandIdx", "normalizedFeatures"], outputCol="Features")
   algo = LogisticRegression(labelCol="Species", featuresCol="Features", maxIter=10, regParam=0.3)
   
   # Chain the steps as stages in a pipeline
   pipeline = Pipeline(stages=[catIndexer, numVector, numScaler, featureVector, algo])
   
   # Use the pipeline to prepare data and fit the model algorithm
   model = pipeline.fit(train)
   print ("Model trained!")
    ```

    由于特征工程步骤现在封装在管道训练的模型中，你可以将模型与测试数据一起使用，而无需应用每个转换（模型会自动应用它们）。

1. 使用以下代码将管道应用于测试数据并对模型进行评估：

    ```python
   prediction = model.transform(test)
   predicted = prediction.select("Features", "probability", col("prediction").astype("Int"), col("Species").alias("trueLabel"))
   display(predicted)

   # Generate evaluation metrics
   from pyspark.ml.evaluation import MulticlassClassificationEvaluator
   
   evaluator = MulticlassClassificationEvaluator(labelCol="Species", predictionCol="prediction")
   
   # Simple accuracy
   accuracy = evaluator.evaluate(prediction, {evaluator.metricName:"accuracy"})
   print("Accuracy:", accuracy)
   
   # Class metrics
   labels = [0,1,2]
   print("\nIndividual class metrics:")
   for label in sorted(labels):
       print ("Class %s" % (label))
   
       # Precision
       precision = evaluator.evaluate(prediction, {evaluator.metricLabel:label,
                                                       evaluator.metricName:"precisionByLabel"})
       print("\tPrecision:", precision)
   
       # Recall
       recall = evaluator.evaluate(prediction, {evaluator.metricLabel:label,
                                                evaluator.metricName:"recallByLabel"})
       print("\tRecall:", recall)
   
       # F1 score
       f1 = evaluator.evaluate(prediction, {evaluator.metricLabel:label,
                                            evaluator.metricName:"fMeasureByLabel"})
       print("\tF1 Score:", f1)
   
   # Weighed (overall) metrics
   overallPrecision = evaluator.evaluate(prediction, {evaluator.metricName:"weightedPrecision"})
   print("Overall Precision:", overallPrecision)
   overallRecall = evaluator.evaluate(prediction, {evaluator.metricName:"weightedRecall"})
   print("Overall Recall:", overallRecall)
   overallF1 = evaluator.evaluate(prediction, {evaluator.metricName:"weightedFMeasure"})
   print("Overall F1 Score:", overallF1) 
    ```

## 注册和部署模型

运行管道时，已经记录了每次试验运行所训练的模型。 你还可以注册** 模型并部署它们，以便将它们提供给客户端应用程序。

> **注意**：模型服务仅在 Azure Databricks 高级** 工作区中受支持，并且仅限于[某些区域](https://learn.microsoft.com/azure/databricks/resources/supported-regions)。

1. 在左窗格中选择“**实验**”。
1. 选择使用笔记本名称生成的试验，并查看最新试验运行的详细信息页。
1. 使用“注册模型”**** 按钮注册已在该试验中记录的模型，并在出现提示时创建一个名为“Penguin Predictor”**** 的新模型。
1. 注册模型后，查看“模型”**** 页面（位于左侧导航栏中）并选择“Penguin Predictor”**** 模型。
1. 在“Penguin Predictor”**** 模型的页面中，通过“使用模型进行推理”**** 按钮创建一个具有以下设置的新实时终结点：
    - **模型**：Penguin Predictor
    - 模型版本****：1
    - 终结点****：predict-penguin
    - 计算大小****：Small

    服务终结点托管在新群集中，创建该群集可能需要几分钟的时间。
  
1. 创建终结点后，请使用右上角的“查询终结点”**** 按钮打开一个界面，你可以在其中测试终结点。 然后在测试界面的“浏览器”**** 选项卡上，输入以下 JSON 请求并使用“发送请求”**** 按钮来调用终结点并生成预测。

    ```json
    {
      "dataframe_records": [
      {
         "Island": "Biscoe",
         "CulmenLength": 48.7,
         "CulmenDepth": 14.1,
         "FlipperLength": 210,
         "BodyMass": 4450
      }
      ]
    }
    ```

1. 对企鹅特征的几个不同值进行试验并观察返回的结果。 然后，关闭测试界面。

## 删除终结点

当不再需要该终结点时，应将其删除，以避免不必要的成本。

在“predict-penguin”**** 终结点页面的“&#8285;”**** 菜单中，选择“删除”****。

## 清理

在 Azure Databricks 门户的“**计算**”页上，选择群集，然后选择“**&#9632; 终止**”以将其关闭。

如果已完成对 Azure Databricks 的探索，则现在可以删除已创建的资源，以避免产生不必要的 Azure 成本并释放订阅中的容量。

> **详细信息**：有关详细信息，请参阅 [Spark MLLib 文档](https://spark.apache.org/docs/latest/ml-guide.html)。
