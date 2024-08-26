# 练习 07 - 使用 Azure Databricks 实现 LLMOps

## 目标
本练习将指导你完成使用 Azure Databricks 实现大型语言模型操作 (LLMOps) 的过程。 在本实验室结束时，你将了解如何使用最佳做法在生产环境中管理、部署和监视大型语言模型 (LLM)。

## 要求
一个有效的 Azure 订阅。 如果没有该帐户，可以注册[免费试用版](https://azure.microsoft.com/en-us/free/)。

## 步骤 1：预配 Azure Databricks
- 登录到 Azure 门户
    1. 转到 Azure 门户，然后使用凭据登录。
- 创建 Databricks 服务：
    1. 导航到“创建资源”>“分析”>“Azure Databricks”。
    2. 输入所需的详细信息，例如工作区名称、订阅、资源组（新建或选择现有），以及位置。
    3. 选择定价层（为此实验室选择标准定价层）。
    4. 单击“查看 + 创建”，然后在通过验证后单击“创建”。

## 步骤 2：启动工作区并创建群集
- 启动 Databricks 工作区
    1. 部署完成后，转到资源并单击“启动工作区”。
- 创建 Spark 群集：
    1. 在 Databricks 工作区中，单击边栏上的“计算”，然后单击“创建计算”。
    2. 指定群集名称并选择 Spark 的运行时版本。
    3. 根据可用选项选择将辅助角色类型作为“标准”和节点类型（选择较小的节点以提高成本效益）。
    4. 单击“创建计算”。

- 安装所需的库
    1. 群集运行后，导航到“库”选项卡。
    2. 安装以下库：
        - azure-ai-openai（用于连接到 Azure OpenAI）
        - mlflow（用于模型管理）
        - scikit-learn（如果需要进行其他模型评估）

## 步骤 3：模型管理
- 上传或访问 LLM
    1. 如果有经过训练的模型，请将其上传到 Databricks 文件系统 (DBFS)，或使用 Azure OpenAI 访问预先训练的模型。
    2. 如果使用 Azure OpenAI

    ```python
    from azure.ai.openai import OpenAIClient

    client = OpenAIClient(api_key="<Your_API_Key>")
    model = client.get_model("gpt-3.5-turbo")

    ```
- 使用 MLflow 对模型进行版本控制
    1. 初始化 MLflow 跟踪

    ```python
    import mlflow

    mlflow.set_tracking_uri("databricks")
    mlflow.start_run()
    ```

- 记录模型

```python
mlflow.pyfunc.log_model("model", python_model=model)
mlflow.end_run()

```

## 步骤 4：模型部署
- 为模型创建 REST API
    1. 为 API 创建 Databricks 笔记本。
    2. 使用 Flask 或 FastAPI 定义 API 终结点

    ```python
    from flask import Flask, request, jsonify
    import mlflow.pyfunc

    app = Flask(__name__)

    @app.route('/predict', methods=['POST'])
    def predict():
        data = request.json
        model = mlflow.pyfunc.load_model("model")
        prediction = model.predict(data["input"])
        return jsonify(prediction)

    if __name__ == '__main__':
        app.run(host='0.0.0.0', port=5000)
    ```
- 保存并运行此笔记本以启动 API。

## 步骤 5：模型监视
- 使用 MLflow 设置日志记录和监视
    1. 在笔记本中启用 MLflow 自动记录

    ```python
    mlflow.autolog()
    ```

    2. 跟踪预测和输入数据。

    ```python
    mlflow.log_param("input", data["input"])
    mlflow.log_metric("prediction", prediction)
    ```

- 实现模型偏移或性能问题的警报
    1. 使用 Azure Databricks 或 Azure Monitor 设置针对模型性能重大更改的警报。

## 步骤 6：模型重新训练和自动化
- 设置自动重新训练管道
    1. 新建用于重新训练的 Databricks 笔记本。
    2. 使用 Databricks 作业或 Azure 数据工厂计划重新训练作业。
    3. 根据数据偏移或时间间隔自动执行重新训练过程。

- 自动部署重新训练的模型
    1. 使用 MLflow 的 model_registry 自动更新已部署的模型。
    2. 使用与步骤 3 中相同的过程部署重新训练的模型。

## 步骤 7：负责任的 AI 做法
- 集成偏差检测和缓解
    1. 使用 Azure 的 Fairlearn 或自定义脚本评估模型偏差。
    2. 使用 MLflow 实现缓解策略并记录结果。

- 实现 LLM 部署的道德准则
    1. 记录输入数据和预测，确保模型预测的透明度。
    2. 建立模型使用指南，并确保符合道德标准。

本练习提供了使用 Azure Databricks 实现 LLMOps 的综合指南，涵盖模型管理、部署、监视、重新训练和负责任的 AI 做法。 执行以下步骤将有助于在生产环境中高效管理和运行 LLM。    