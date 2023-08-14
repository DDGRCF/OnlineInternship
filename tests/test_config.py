import os
import sys
import unittest

CHATBOT_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "chatbot")
sys.path.append(CHATBOT_DIR)

from util import Config


class TestConfig(unittest.TestCase):

    def test_config_str(self):
        config = Config()
        self.assertTrue(True)
        print(config)


if __name__ == "__main__":
    unittest.main()
