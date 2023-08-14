import os
import sys
import unittest

CHATBOT_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "chatbot")
sys.path.append(CHATBOT_DIR)

from util import ModelType, LLaMA2


class TestModel(unittest.TestCase):

    def test_model_type(self):
        self.assertEqual(ModelType.get_type("llama2"), ModelType.LLAMA2)
        self.assertEqual(ModelType.get_model("llama2"), LLaMA2)

    def test_llama2_cls(self):
        llama2 = LLaMA2()
        prompt = list(LLaMA2.get_prompt("Hi do you know Pytorch?"))
        self.assertTrue(len(prompt) > 1)

if __name__ == "__main__":
    unittest.main()
