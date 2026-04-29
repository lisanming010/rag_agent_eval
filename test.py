# from tool.config_reader import ConfigReader

# test = ConfigReader.get_instance().get("judge_thresholds.contextual_recall")
# print(test)
# print(type(test))

import os
filename = 'test_suite/test_cases.csv'
print(os.path.splitext(filename)[0])