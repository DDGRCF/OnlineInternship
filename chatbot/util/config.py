class Config:
    def __init__(self, *args, **kwargs):
        self.temperature = 0.1
        self.top_p = 0.9
        self.max_seq_len = 512
        self.pre_prompt = "You are a code comment assistant. For any language code you input, first explain the use of this code, and then comment it in Chinese, without any other words except this."



