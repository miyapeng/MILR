import re
from termcolor import colored
import torch

class RewardModel(object):
    def __init__(
            self, 
            model,
            tokenizer,
            data_name: str = "gsm8k", 
            device: str = "cuda",
            rule_format_string: str = None,
        ):
        """
        Args:
            model: the model to use for reward prediction
            tokenizer: the tokenizer to use for reward prediction
            data_name
            device
            rule_format_string: str, the answer format that the solution should follow
        """

        self.model = model
        self.tokenizer = tokenizer

        self.type = type

        self.data_name = data_name 
        self.device = device
        self.rule_format_string = rule_format_string 



        
    def get_reward(self, question, solution):
        '''
        Get reward from question and solution.

        Args:
            question: str, question
            solution: str, solution
        Returns:
            reward: int, reward
        '''
        # TODO
