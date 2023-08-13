# OnlineInternship

## 简介

移动线上实习任务

## 说明

1. 使用前端：streamlit
2. 小组分工：
    * 总体框架书写: R
    * 输入一个代码，产生注释：Yue
    * 具有上下文的代码注释
      (输入文件夹，把文件夹里面的代码文件内容输入到大模型，再输入要解释的代码)：mamianshusheng
3. git 提交代码
    * commit风格:
      * feat: 新功能
      * fix: 解决bug
      * docs: 文件解释

## 使用

1. 创建虚拟环境

    ```shell
    conda create -n online-internship python=3.9 -y
    ```

2. 安装依赖

    ```shell
    pip install -r requirements.txt
    pip install -U streamlit
    ```

3. 运行程序

    ```shell
    streamlit run chatbot/chatbot.py
    ```

    **Note**：当修改程序后，不需要重新启动，只需要刷新页面就可以热更新

## 要点

