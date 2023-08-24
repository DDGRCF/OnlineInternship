class Config:
    def __init__(self, *args, **kwargs):
        self.temperature = 0.1
        self.top_p = 0.9
        self.top_k = 40
        self.max_seq_len = 2048
        self.system_prompt = "You are a professional, efficient and knowledgeable code helper. For any code entered, you should add adequate, clear comments to the code that help to understand the function of the code. Your responses should still maintain the original typographical style of the code. After the comments have been added, you should explain the purpose of the code."
        self.log_dir = ""
        self.model = "LLaMA2"
        self.model_path = ""

    def __str__(self):
        return ""

