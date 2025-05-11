# DeepSeek PDF 问答系统

这是一个基于LangChain和FAISS的PDF文档问答系统，可以从PDF文件中提取信息，建立向量数据库，并回答用户关于文档内容的问题。

## 功能特点

- 支持处理多个PDF文件
- 使用FAISS向量数据库高效存储和检索文本
- 基于大语言模型实现准确的问答功能
- 显示答案来源的文件名和页码
- 交互式命令行界面，支持连续提问
- 自动保存向量数据库，避免重复处理

## 安装依赖

确保你已经安装了Python 3.8或更高版本，然后安装以下依赖：

```bash
pip install PyPDF2 langchain langchain_community langchain_openai faiss-cpu dashscope
```

## 设置API密钥

在使用本系统前，你需要设置DashScope API密钥。请在代码中找到以下行：

```python
# DashScope API密钥
DASHSCOPE_API_KEY = 'sk-'
```

将其修改为你的有效DashScope API密钥：

```python
DASHSCOPE_API_KEY = 'sk-your_api_key_here'
```

> **重要提示**：你可以在[DashScope官网](https://dashscope.aliyun.com/)注册账号并获取API密钥。

## 使用方法

1. 准备你的PDF文件，并在`main`函数中更新PDF文件路径列表：

```python
pdf_paths = [
    '清华大学104页《DeepSeek：从入门到精通》.pdf',
    'DeepSeek指导手册(24页).pdf'
]
```

2. 运行程序：

```bash
python chatpdf-faiss.py
```

3. 程序将处理PDF文件，创建向量数据库，然后进入交互式问答界面
4. 输入你的问题，程序会基于PDF内容回答
5. 每个回答后会显示信息来源（文件名和页码）
6. 输入`q`退出程序

## 高级用法

### 加载已保存的向量数据库

如果你已经处理过PDF并保存了向量数据库，可以直接加载它而不需要重新处理PDF文件：

```python
# 取消注释以下代码以加载已保存的向量数据库
embeddings = DashScopeEmbeddings(
    model="text-embedding-v1",
    dashscope_api_key=DASHSCOPE_API_KEY,
)
loaded_knowledge_base = load_knowledge_base("./vector_db", embeddings)
process_query(loaded_knowledge_base, "你的问题")
```

### 调整参数

你可以根据需要调整以下参数：

- 文本分割：修改`RecursiveCharacterTextSplitter`的参数
- 搜索相关度：修改`similarity_search`中的`k`参数
- 语言模型：修改`Tongyi`的`model_name`参数

## 注意事项

- 处理大型PDF文件可能需要较长时间
- API调用会产生费用，请合理使用
- 向量数据库会占用磁盘空间，可能需要定期清理

## 常见问题

- **Q: 为什么程序找不到我的PDF文件？**  
  A: 请确保PDF文件路径正确，可以使用绝对路径避免问题。

- **Q: 为什么回答不准确？**  
  A: 可能是PDF内容质量问题或文本分割不合理，可以尝试调整参数。

- **Q: 程序报错"API密钥无效"？**  
  A: 请确保你已设置正确的DashScope API密钥。