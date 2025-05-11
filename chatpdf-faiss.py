#!/usr/bin/env python
# coding: utf-8

import os
import pickle
from typing import List, Tuple

from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.callbacks.manager import get_openai_callback
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.llms import Tongyi
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# DashScope API密钥
DASHSCOPE_API_KEY = 'sk-'


def extract_text_with_page_numbers(pdf) -> Tuple[str, List[int]]:
    """
    从PDF中提取文本并记录每行文本对应的页码
    
    参数:
        pdf: PDF文件对象
    
    返回:
        text: 提取的文本内容
        page_numbers: 每行文本对应的页码列表
    """
    text = ""
    page_numbers = []

    for page_number, page in enumerate(pdf.pages, start=1):
        extracted_text = page.extract_text()
        if extracted_text:
            text += extracted_text
            page_numbers.extend([page_number] * len(extracted_text.split("\n")))
        else:
            print(f"警告: 在第 {page_number} 页未找到文本。")

    return text, page_numbers


def process_multiple_pdfs(pdf_paths):
    """
    处理多个PDF文件，提取文本和页码信息
    
    参数:
        pdf_paths: PDF文件路径列表
    
    返回:
        combined_text: 合并后的文本内容
        combined_page_info: 合并后的页码和文件名信息列表
    """
    combined_text = ""
    combined_page_info = []
    
    for pdf_path in pdf_paths:
        print(f"正在处理PDF文件: {pdf_path}")
        pdf_reader = PdfReader(pdf_path)
        text, page_numbers = extract_text_with_page_numbers(pdf_reader)
        combined_text += f"\n\n=== 文件: {pdf_path} ===\n\n{text}"
        
        # 为每个页码添加文件名信息
        file_page_info = [{"file": pdf_path, "page": page} for page in page_numbers]
        combined_page_info.extend(file_page_info)
        
        print(f"文件 {pdf_path} 提取的文本长度: {len(text)} 个字符")
    
    return combined_text, combined_page_info


def process_text_with_splitter(text: str, page_info: List[dict], save_path: str = None) -> FAISS:
    """
    处理文本并创建向量存储
    
    参数:
        text: 提取的文本内容
        page_info: 每行文本对应的页码和文件名信息列表
        save_path: 可选，保存向量数据库的路径
    
    返回:
        knowledgeBase: 基于FAISS的向量存储对象
    """
    # 创建文本分割器，用于将长文本分割成小块
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", " ", ""],
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )

    # 分割文本
    chunks = text_splitter.split_text(text)
    print(f"文本被分割成 {len(chunks)} 个块。")
        
    # 创建嵌入模型
    embeddings = DashScopeEmbeddings(
        model="text-embedding-v1",
        dashscope_api_key=DASHSCOPE_API_KEY,
    )
    
    # 从文本块创建知识库
    knowledgeBase = FAISS.from_texts(chunks, embeddings)
    print("已从文本块创建知识库。")
    
    # 存储每个文本块对应的页码和文件信息
    chunk_info = {chunk: page_info[i] for i, chunk in enumerate(chunks)}
    knowledgeBase.chunk_info = chunk_info
    
    # 如果提供了保存路径，则保存向量数据库和页码信息
    if save_path:
        # 确保目录存在
        os.makedirs(save_path, exist_ok=True)
        
        # 保存FAISS向量数据库
        knowledgeBase.save_local(save_path)
        print(f"向量数据库已保存到: {save_path}")
        
        # 保存页码和文件信息到同一目录
        with open(os.path.join(save_path, "chunk_info.pkl"), "wb") as f:
            pickle.dump(chunk_info, f)
        print(f"文档块信息已保存到: {os.path.join(save_path, 'chunk_info.pkl')}")

    return knowledgeBase


def load_knowledge_base(load_path: str, embeddings = None) -> FAISS:
    """
    从磁盘加载向量数据库和页码信息
    
    参数:
        load_path: 向量数据库的保存路径
        embeddings: 可选，嵌入模型。如果为None，将创建一个新的DashScopeEmbeddings实例
    
    返回:
        knowledgeBase: 加载的FAISS向量数据库对象
    """
    # 如果没有提供嵌入模型，则创建一个新的
    if embeddings is None:
        embeddings = DashScopeEmbeddings(
            model="text-embedding-v1",
            dashscope_api_key=DASHSCOPE_API_KEY,
        )
    
    # 加载FAISS向量数据库，添加allow_dangerous_deserialization=True参数以允许反序列化
    knowledgeBase = FAISS.load_local(load_path, embeddings, allow_dangerous_deserialization=True)
    print(f"向量数据库已从 {load_path} 加载。")
    
    # 加载页码和文件信息
    chunk_info_path = os.path.join(load_path, "chunk_info.pkl")
    if os.path.exists(chunk_info_path):
        with open(chunk_info_path, "rb") as f:
            chunk_info = pickle.load(f)
        knowledgeBase.chunk_info = chunk_info
        print("文档块信息已加载。")
    else:
        # 尝试加载旧版本的页码信息
        page_info_path = os.path.join(load_path, "page_info.pkl")
        if os.path.exists(page_info_path):
            with open(page_info_path, "rb") as f:
                page_info = pickle.load(f)
            knowledgeBase.page_info = page_info
            print("旧版本页码信息已加载。")
        else:
            print("警告: 未找到文档块信息文件。")
    
    return knowledgeBase


def process_query(knowledge_base, query_text):
    """
    处理用户查询并返回结果
    
    参数:
        knowledge_base: 向量数据库对象
        query_text: 查询文本
    """
    # 创建大语言模型
    llm = Tongyi(model_name="deepseek-v3", dashscope_api_key=DASHSCOPE_API_KEY)
    
    # 执行相似度搜索，找到与查询相关的文档
    docs = knowledge_base.similarity_search(query_text, k=10)

    # 定义问答提示模板
    template = """使用以下上下文片段回答问题。如果你不知道答案，只需说"我不知道"，不要试图编造答案。

上下文: {context}

问题: {question}

答案:"""
    
    prompt = PromptTemplate.from_template(template)
    
    # 设置文档处理函数，将多个文档合并成一个字符串
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    # 使用LangChain表达语言(LCEL)创建问答链
    rag_chain = (
        {"context": lambda x: format_docs(x["documents"]), "question": lambda x: x["question"]}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    # 使用回调函数跟踪API调用成本
    with get_openai_callback() as cost:
        # 执行问答链
        response = rag_chain.invoke({"documents": docs, "question": query_text})
        print(f"查询已处理。成本: {cost}")
        print(response)
        print("\n参考来源:")

    # 记录唯一的页码和文件信息
    unique_sources = set()

    # 显示每个文档块的来源信息
    for doc in docs:
        text_content = getattr(doc, "page_content", "")
        
        # 检查是否使用新的chunk_info格式
        if hasattr(knowledge_base, 'chunk_info'):
            source_info = knowledge_base.chunk_info.get(text_content.strip())
            if source_info:
                source_key = f"{source_info['file']}:{source_info['page']}"
                if source_key not in unique_sources:
                    unique_sources.add(source_key)
                    print(f"文件: {source_info['file']} - 页码: {source_info['page']}")
        # 兼容旧版本
        elif hasattr(knowledge_base, 'page_info'):
            source_page = knowledge_base.page_info.get(text_content.strip(), "未知")
            if source_page not in unique_sources:
                unique_sources.add(source_page)
                print(f"页码: {source_page}")
        else:
            print("警告: 未找到文档块来源信息。")


def main():
    """主函数，协调整个流程的执行"""
    # 定义要处理的PDF文件路径列表
    pdf_paths = [
        '清华大学104页《DeepSeek：从入门到精通》.pdf',
        'DeepSeek指导手册(24页).pdf'
    ]

    # 处理所有PDF文件
    text, page_info = process_multiple_pdfs(pdf_paths)
    print(f"提取的文本总长度: {len(text)} 个字符。")
    
    # 处理文本并创建知识库，同时保存到磁盘
    save_dir = "./vector_db"
    knowledge_base = process_text_with_splitter(text, page_info, save_path=save_dir)
    
    # 交互式问答循环
    print("\n=== DeepSeek PDF 问答系统 ===")
    print("输入您的问题，或输入 'q' 退出")
    
    while True:
        # 获取用户输入
        user_query = input("\n请输入您的问题: ")
        
        # 检查是否退出
        if user_query.lower() in ['q', 'quit', 'exit', '退出']:
            print("感谢使用，再见！")
            break
        
        # 如果输入为空，继续循环
        if not user_query.strip():
            print("问题不能为空，请重新输入。")
            continue
        
        # 处理用户查询
        print(f"\n处理查询: {user_query}")
        process_query(knowledge_base, user_query)
        
        print("\n--- 输入新问题或输入 'q' 退出 ---")
    
    # 加载已保存的向量数据库的示例
    # 要运行此代码，请取消下面的注释
    """
    # 创建嵌入模型
    embeddings = DashScopeEmbeddings(
        model="text-embedding-v1",
        dashscope_api_key=DASHSCOPE_API_KEY,
    )
    # 从磁盘加载向量数据库
    loaded_knowledge_base = load_knowledge_base("./vector_db", embeddings)
    # 使用加载的知识库进行查询
    process_query(loaded_knowledge_base, "DeepSeek模型有哪些特点？")
    """


if __name__ == "__main__":
    main()

