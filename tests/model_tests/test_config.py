import unittest, os
from definitions import CONFIG_PATH

from model.config import ClaudeConfig

class TestClaudeConfig(unittest.TestCase):

    def test_yaml_load(self):
        # Arrange (Set Up Test)
        defaultClaudeConfigData = open(os.path.join(CONFIG_PATH, "DefaultClaudeConfig.yaml"))
        # Act (Perform Action to be Tested)
        resultClaudeConfig = ClaudeConfig.load_from_yaml(defaultClaudeConfigData)
        # Assert (Check our results)
        print(resultClaudeConfig)