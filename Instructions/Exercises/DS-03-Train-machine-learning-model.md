---
lab:
  title: Azure Databricks 中的机器学习入门
---

# Azure Databricks 中的机器学习入门

在本练习中，你将探索在 Azure Databricks 中准备数据和训练机器学习模型的技术。

完成此练习大约需要 45 分钟。

## 准备工作

需要一个你在其中具有管理级权限的 [Azure 订阅](https://azure.microsoft.com/free)。

## 预配 Azure Databricks 工作区

> **提示**：如果你已有 Azure Databricks 工作区，则可以跳过此过程并使用现有工作区。

本练习包括一个用于预配新 Azure Databricks 工作区的脚本。 该脚本会尝试在一个区域中创建*高级*层 Azure Databricks 工作区资源，在该区域中，Azure 订阅具有本练习所需计算核心的充足配额；该脚本假设你的用户帐户在订阅中具有足够的权限来创建 Azure Databricks 工作区资源。 如果脚本由于配额或权限不足失败，可以尝试 [在 Azure 门户中以交互方式创建 Azure Databricks 工作区](https://learn.microsoft.com/azure/databricks/getting-started/#--create-an-azure-databricks-workspace)。

1. 在 Web 浏览器中，登录到 [Azure 门户](https://portal.azure.com)，网址为 `https://portal.azure.com`。
2. 使用页面顶部搜索栏右侧的 [\>_] 按钮在 Azure 门户中创建新的 Cloud Shell，在出现提示时选择“PowerShell”环境并创建存储。 Cloud Shell 在 Azure 门户底部的窗格中提供命令行界面，如下所示：

    ![具有 Cloud Shell 窗格的 Azure 门户](./images/cloud-shell.png)

    > 注意：如果以前创建了使用 Bash 环境的 Cloud shell，请使用 Cloud Shell 窗格左上角的下拉菜单将其更改为“PowerShell”。

3. 请注意，可以通过拖动窗格顶部的分隔条或使用窗格右上角的 &#8212;、&#9723; 或 X 图标来调整 Cloud Shell 的大小，以最小化、最大化和关闭窗格  。 有关如何使用 Azure Cloud Shell 的详细信息，请参阅 [Azure Cloud Shell 文档](https://docs.microsoft.com/azure/cloud-shell/overview)。

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

## 引入数据

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

## 浏览和清理数据
  
引入数据文件后，即可将其加载到数据帧中并查看它。

1. 在现有代码单元格下，使用 + 图标添加新的代码单元格****。 然后在新单元格中输入并运行以下代码，以从文件加载数据并显示它。

    ```python
   df = spark.read.format("csv").option("header", "true").load("/ml_lab/penguins.csv")
   display(df)
    ```

    该代码会启动必要的 *Spark 作业*来加载数据，输出是名为 *df* 的 *pyspark.sql.dataframe.DataFrame* 对象。 你将看到此信息直接显示在代码下，可以使用 **&#9656;** 开关来展开 **df: pyspark.sql.dataframe.DataFrame** 输出，并查看它包含的列及其数据类型的详细信息。 由于此数据是从文本文件加载的并包含一些空白值，因此 Spark 已将**字符串**数据类型分配给所有列。
    
    数据本身包含对南极洲观察到的企鹅的以下详细信息的测量：
    
    - **Island**：在南极洲观察到企鹅的岛屿。
    - **CulmenLength**：企鹅的喙的长度（毫米）。
    - **CulmenDepth**：企鹅的喙的深度（毫米）。
    - **FlipperLength**：企鹅鳍状肢的长度（毫米）。
    - **BodyMass**：企鹅的体重（克）。
    - **Species**：一个整数值，表示企鹅的物种：
      - **0**：** 阿德利企鹅
      - **1**：白眉企鹅**
      - **2**：帽带企鹅**
      
    这个项目的目标是使用企鹅的观测特征（其*特征*）来预测其物种（在机器学习术语中，我们称之为*标签*）。
      
    请注意，某些观察值包含某些特征的 *null* 或“缺失”数据值。 你引入的原始源数据存在这样的问题并不少见，因此通常机器学习项目中的第一个阶段是彻底浏览数据并对其进行清理，使其更适合训练机器学习模型。
    
1. 添加一个单元格并使用它运行以下单元格，以使用 **dropna** 方法删除具有不完整数据的行，并使用 **col** 和 **astype** 函数通过 **select** 方法向数据应用适当的数据类型。

    ```python
   from pyspark.sql.types import *
   from pyspark.sql.functions import *
   
   data = df.dropna().select(col("Island").astype("string"),
                              col("CulmenLength").astype("float"),
                             col("CulmenDepth").astype("float"),
                             col("FlipperLength").astype("float"),
                             col("BodyMass").astype("float"),
                             col("Species").astype("int")
                             )
   display(data)
    ```
    
    可以再次切换返回的数据帧的详细信息（这次命名为*数据*）以验证数据类型是否已应用，并且可以查看数据以验证包含不完整数据的行是否已删除。
    
    在实际项目中，可能需要执行更多浏览和数据清理来修复数据中的错误，识别和删除离群值（过大或过小的值），或者平衡数据，使得你尝试预测的每个标签的行数相当。

    > **提示**：可以在 [Spark SQL 参考](https://spark.apache.org/docs/latest/sql-programming-guide.html)中了解有关可用于数据帧的方法和函数的详细信息。

## 拆分数据

在本练习中，我们假定数据现在已得到适当的清理，可供我们用来训练机器学习模型。 我们将尝试预测的标签是特定的类别或*类*（企鹅的物种），因此我们需要训练的机器学习模型的类型是*分类*模型。 分类（以及用于预测数值的*回归*）是一种*监督式*机器学习形式，其中我们使用的训练数据包含要预测的标签的已知值。 训练模型的过程实际上只是将算法拟合到数据，以计算特征值如何与已知标签值关联。 然后，我们可以将训练后的模型应用于一个我们只知道特征值的新观察，并让它预测标签值。

为了确保我们对训练后的模型有信心，典型的方法是仅使用*一部分*数据来训练模型，并保留一些具有已知标签值的数据，用于测试训练后的模型，检查它预测的准确性。 为了实现此目标，我们将完整数据集拆分为两个随机子集。 我们将使用 70% 的数据进行训练，并保留 30% 进行测试。

1. 添加并运行包含以下代码的代码单元以拆分数据。

    ```python
   splits = data.randomSplit([0.7, 0.3])
   train = splits[0]
   test = splits[1]
   print ("Training Rows:", train.count(), " Testing Rows:", test.count())
    ```

## 执行特征工程

清理原始数据后，数据科学家通常会执行一些额外的工作来准备模型训练。 此过程通常称为*特征工程*，它涉及迭代优化训练数据集中的特征以生成最佳模型。 所需的特定特征修改取决于数据和所需模型，但你应该熟悉一些常见的特征工程任务。

### 编码分类特征

机器学习算法通常基于查找特征和标签之间的数学关系。 这意味着，通常最好将训练数据中的特征定义为*数值*。 在某些情况下，你可能有一些特征属于*分类*而非数值，这些特征以字符串表示，例如，在数据集中出现的观察到企鹅的岛屿名称就是这样。 但是，大多数算法需要数值特征，因此，这些基于字符串的分类值需要*编码*为数字。 在这种情况下，我们将使用 **Spark MLLib** 库中的 **StringIndexer** 将岛名称编码为数值，方法是为每个离散的岛屿名称分配唯一的整数索引。

1. 运行以下代码，将 **Island** 分类列值编码为数值索引。

    ```python
   from pyspark.ml.feature import StringIndexer

   indexer = StringIndexer(inputCol="Island", outputCol="IslandIdx")
   indexedData = indexer.fit(train).transform(train).drop("Island")
   display(indexedData)
    ```

    在结果中，你不会再看到岛屿名称，而应该看到每行现在都有一个 **IslandIdx** 列，其中包含一个整数值，表示观察记录所在的岛屿。

### 规范化（刻度）数值特征

现在，让我们关注数据中的数值。 这些值（**CulmenLength**、**CulmenDepth**、**FlipperLength** 和 **BodyMass**）都表示一种或另一种度量值，但它们使用着不同的刻度。 训练模型时，度量单位没有不同观察值之间的相对差异重要，而由较大数字表示的特征通常主宰着模型训练算法，从而扭曲计算预测时特征的重要性。 为了缓解这种情况，通常会*规范化*数值特征值，使它们都处于相同的相对刻度上（例如，介于 0.0 和 1.0 之间的十进制值）。

我们将用于执行此操作的代码比我们之前所做的分类编码要复杂一点。 我们需要同时缩放多个列值，因此我们使用的方法是创建包含所有数值特征的*矢量*（实质上是数组）的单个列，然后应用缩放器来生成具有等效规范化值的新矢量列。

1. 使用以下代码规范化数值特征，并查看预规范化和规范化矢量列的比较。

    ```python
   from pyspark.ml.feature import VectorAssembler, MinMaxScaler

   # Create a vector column containing all numeric features
   numericFeatures = ["CulmenLength", "CulmenDepth", "FlipperLength", "BodyMass"]
   numericColVector = VectorAssembler(inputCols=numericFeatures, outputCol="numericFeatures")
   vectorizedData = numericColVector.transform(indexedData)
   
   # Use a MinMax scaler to normalize the numeric values in the vector
   minMax = MinMaxScaler(inputCol = numericColVector.getOutputCol(), outputCol="normalizedFeatures")
   scaledData = minMax.fit(vectorizedData).transform(vectorizedData)
   
   # Display the data with numeric feature vectors (before and after scaling)
   compareNumerics = scaledData.select("numericFeatures", "normalizedFeatures")
   display(compareNumerics)
    ```

    结果中的 **numericFeatures** 列包含每行的矢量。 该矢量包含四个未缩放的数值（企鹅的原始度量值）。 可以使用 **&#9656;** 开关更清楚地看到离散值。
    
    **normalizedFeatures** 列还包含每个企鹅观察的矢量，但这次，矢量中的值根据每个度量的最小值和最大值规范化为相对刻度。

### 准备用于训练的特征和标签

现在，让我们将所有内容组合在一起，创建一个包含所有特征的列（编码后的分类岛屿名称和规范化的企鹅测量值），另一列包含我们要训练模型进行预测的类标签（企鹅物种）。

1. 运行以下代码：

    ```python
   featVect = VectorAssembler(inputCols=["IslandIdx", "normalizedFeatures"], outputCol="featuresVector")
   preppedData = featVect.transform(scaledData)[col("featuresVector").alias("features"), col("Species").alias("label")]
   display(preppedData)
    ```

    **特征**矢量包含五个值（编码后的岛屿和规范化的喙长度、喙深度、鳍状肢长度和体重）。 该标签包含一个简单的整数代码，指示企鹅物种的类。

## 训练机器学习模型

准备好训练数据后，即可使用它来训练模型。 模型训练使用的*算法*会尝试在特征和标签之间建立关系。 由于在这种情况下，你需要训练预测*类*的类别的模型，因此需要使用*分类*算法。 分类有许多算法 - 让我们先讨论其中一个完善的算法：逻辑回归，它会迭代地尝试查找可应用于逻辑计算中特征数据的最佳系数，以预测每个类标签值的概率。 若要训练该模型，需要将逻辑回归算法拟合到训练数据。

1. 运行以下代码来训练模型。

    ```python
   from pyspark.ml.classification import LogisticRegression

   lr = LogisticRegression(labelCol="label", featuresCol="features", maxIter=10, regParam=0.3)
   model = lr.fit(preppedData)
   print ("Model trained!")
    ```

    大多数算法都支持参数，这些参数可让你控制模型训练的方式。 在这种情况下，逻辑回归算法需要你标识包含特征矢量的列和包含已知标签的列，它还让你可以指定最多执行几次迭代来查找逻辑计算的最佳系数，以及用于防止模型*过度拟合*的正则化参数（换句话说，建立适用于训练数据的逻辑计算，但在应用于新数据时不会很好地通用化）。

## 测试模型

现在，你已经训练了模型，可以使用你保留的数据测试它了。 在执行此操作之前，需要将用于训练数据的相同特征工程转换在测试数据上执行（在本例中，即编码岛屿名称并规范化度量值）。 然后，可以使用模型来预测测试数据中特征的标签，并将预测的标签与实际已知的标签进行比较。

1. 使用以下代码准备测试数据，然后生成预测：

    ```python
   # Prepare the test data
   indexedTestData = indexer.fit(test).transform(test).drop("Island")
   vectorizedTestData = numericColVector.transform(indexedTestData)
   scaledTestData = minMax.fit(vectorizedTestData).transform(vectorizedTestData)
   preppedTestData = featVect.transform(scaledTestData)[col("featuresVector").alias("features"), col("Species").alias("label")]
   
   # Get predictions
   prediction = model.transform(preppedTestData)
   predicted = prediction.select("features", "probability", col("prediction").astype("Int"), col("label").alias("trueLabel"))
   display(predicted)
    ```

    结果包含以下列：
    
    - **特征**：测试数据集中准备好的特征数据。
    - **概率**：由模型计算的每个类的概率。 这由包含三个概率值的矢量组成（因为有三个类），总共加起来为 1.0（它假定企鹅属于三个物种类*之一*的概率为 100%）。
    - **预测**：预测的类标签（概率最高的那个）。
    - **trueLabel**：测试数据中的实际已知标签值。
    
    若要评估模型的有效性，只需比较这些结果中的预测和真实标签。 但是，可以使用模型评估器获取更有意义的指标 - 在这种情况下，可以使用多类（因为有多个可能的类标签）分类评估器。

1. 使用以下代码根据测试数据的结果获取分类模型的评估指标：

    ```python
   from pyspark.ml.evaluation import MulticlassClassificationEvaluator
   
   evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction")
   
   # Simple accuracy
   accuracy = evaluator.evaluate(prediction, {evaluator.metricName:"accuracy"})
   print("Accuracy:", accuracy)
   
   # Individual class metrics
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
   
   # Weighted (overall) metrics
   overallPrecision = evaluator.evaluate(prediction, {evaluator.metricName:"weightedPrecision"})
   print("Overall Precision:", overallPrecision)
   overallRecall = evaluator.evaluate(prediction, {evaluator.metricName:"weightedRecall"})
   print("Overall Recall:", overallRecall)
   overallF1 = evaluator.evaluate(prediction, {evaluator.metricName:"weightedFMeasure"})
   print("Overall F1 Score:", overallF1)
    ```

    为多类分类计算的评估指标包括：
    
    - 准确性****：正确预测的总体比例。
    - 按类指标：
      - 精准率****：此类的正确预测的比例。
      - 召回率****：此类的正确预测的实际实例的比例。
      - **F1 分数**：精准率和召回率的组合指标
    - 所有类的组合（加权）精准率、召回率和 F1 指标。
    
    > **注意**：最初看起来可能总体准确性指标提供了评估模型预测性能的最佳方法。 但是，请考虑这一点。 假设巴布亚企鹅在你的研究地点占企鹅总数的 95%。 始终预测标签 **1**（巴布亚类）的模型的准确性为 0.95。 这并不意味着它是基于特征预测企鹅物种的好模型！ 这就是为什么数据科学家倾向于探索其他指标，以更好地了解分类模型在预测每个可能的类标签方面的表现。

## 使用管道

执行所需的特征工程步骤，然后将算法拟合到数据，由此来训练模型。 若要将模型与某些测试数据一起使用来生成预测（称为*推理*），则必须对测试数据应用相同的特征工程步骤。 一种生成和使用模型的更高效方法是封装用于准备数据的转换器，以及用于在*管道*中训练它的模型。

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

1. 使用以下代码将管道应用于测试数据：

    ```python
   prediction = model.transform(test)
   predicted = prediction.select("Features", "probability", col("prediction").astype("Int"), col("Species").alias("trueLabel"))
   display(predicted)
    ```

## 尝试其他算法

到目前为止，你已使用逻辑回归算法训练了分类模型。 让我们更改管道中的该阶段以尝试其他算法。

1. 运行以下代码以创建使用决策树算法的管道：

    ```python
   from pyspark.ml import Pipeline
   from pyspark.ml.feature import StringIndexer, VectorAssembler, MinMaxScaler
   from pyspark.ml.classification import DecisionTreeClassifier
   
   catFeature = "Island"
   numFeatures = ["CulmenLength", "CulmenDepth", "FlipperLength", "BodyMass"]
   
   # Define the feature engineering and model steps
   catIndexer = StringIndexer(inputCol=catFeature, outputCol=catFeature + "Idx")
   numVector = VectorAssembler(inputCols=numFeatures, outputCol="numericFeatures")
   numScaler = MinMaxScaler(inputCol = numVector.getOutputCol(), outputCol="normalizedFeatures")
   featureVector = VectorAssembler(inputCols=["IslandIdx", "normalizedFeatures"], outputCol="Features")
   algo = DecisionTreeClassifier(labelCol="Species", featuresCol="Features", maxDepth=10)
   
   # Chain the steps as stages in a pipeline
   pipeline = Pipeline(stages=[catIndexer, numVector, numScaler, featureVector, algo])
   
   # Use the pipeline to prepare data and fit the model algorithm
   model = pipeline.fit(train)
   print ("Model trained!")
    ```

    这一次，管道包含与之前相同的特征准备阶段，但使用*决策树*算法来训练模型。
    
   1. 运行以下代码将新管道用于测试数据：

    ```python
   # Get predictions
   prediction = model.transform(test)
   predicted = prediction.select("Features", "probability", col("prediction").astype("Int"), col("Species").alias("trueLabel"))
   
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

## 保存模型

实际上，你会迭代地尝试使用不同的算法（和参数）训练模型，从而找到最适合你的数据模型。 现在，我们将继续使用经过训练的决策树模型。 让我们保存它，以便稍后可以将它用于一些新观察到的企鹅。

1. 使用以下代码保存模型：

    ```python
   model.save("/models/penguin.model")
    ```

    现在，当你出去发现新的企鹅时，你可以加载保存的模型，根据你对企鹅特征的测量使用该模型来预测它们的物种。 使用模型从新数据生成预测的过程称为*推理*。

1. 运行以下代码来加载模型，并使用它来预测新观察到的企鹅的物种：

    ```python
   from pyspark.ml.pipeline import PipelineModel

   persistedModel = PipelineModel.load("/models/penguin.model")
   
   newData = spark.createDataFrame ([{"Island": "Biscoe",
                                     "CulmenLength": 47.6,
                                     "CulmenDepth": 14.5,
                                     "FlipperLength": 215,
                                     "BodyMass": 5400}])
   
   
   predictions = persistedModel.transform(newData)
   display(predictions.select("Island", "CulmenDepth", "CulmenLength", "FlipperLength", "BodyMass", col("prediction").alias("PredictedSpecies")))
    ```

## 清理

在 Azure Databricks 门户的“**计算**”页上，选择群集，然后选择“**&#9632; 终止**”以将其关闭。

如果已完成对 Azure Databricks 的探索，则现在可以删除已创建的资源，以避免产生不必要的 Azure 成本并释放订阅中的容量。

> **详细信息**：有关详细信息，请参阅 [Spark MLLib 文档](https://spark.apache.org/docs/latest/ml-guide.html)。
