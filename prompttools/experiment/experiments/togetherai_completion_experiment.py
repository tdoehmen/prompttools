# Copyright (c) MotherDuck, Inc.
# All rights reserved.
#
# This source code's license can be found in the
# LICENSE file in the root directory of this source tree.

import os
from typing import Dict, List, Optional, Union
import together

from prompttools.selector.prompt_selector import PromptSelector
from .experiment import Experiment


class TogetherAICompletionExperiment(Experiment):
    r"""
    This class defines an experiment for TogetherAI's completion API.
    It accepts lists for each argument passed into TogetherAI's API, then creates
    a cartesian product of those arguments, and gets results for each.

    Note:
        - All arguments here should be a ``list``, even if you want to keep the argument frozen
          (i.e. ``temperature=[1.0]``), because the experiment will try all possible combination
          of the input arguments.
        - For detailed description of the input arguments, please reference at TogetherAI's completion API.

    Args:
        model (list[str]): list of ID(s) of the model(s) to use, e.g. ``["gpt-3.5-turbo", "ft:gpt-3.5-turbo:org_id"]``
            If you are using Azure TogetherAI service, put the models' deployment names here

        prompt (list[str]): the prompt(s) to generate completions for, encoded as a string, array of strings,
            array of tokens, or array of token arrays.

        max_tokens (list[int]):
            Defaults to [128]. The maximum number of tokens to generate in the chat completion.

        stop (list[list[str]]):
            Defaults to ["<human>"]. List of stop words the model should stop generation at.

        temperature (list[float]):
            Defaults to [0.7]. A decimal number that determines the degree of randomness in the response.

        top_p (list[float]):
            Defaults to [0.7]. Used to dynamically adjust the number of choices for each predicted token based
            on the cumulative probabilities. A value of 1 will always yield the same output.
            A temperature less than 1 favors more correctness and is appropriate for question answering or
            summarization. A value greater than 1 introduces more randomness in the output.

        top_k (list[int]):
            Defaults to [50]. Used to limit the number of choices for the next predicted word or token.
            It specifies the maximum number of tokens to consider at each step, based on their probability
            of occurrence. This technique helps to speed up the generation process and can improve the
            quality of the generated text by focusing on the most likely options.

        repetition_penalty (list[float]):
            Defaults to [1.0]. A number that controls the diversity of generated text by reducing the likelihood
            of repeated sequences. Higher values decrease repetition.

        logprobs (list[int]):
            Defaults to [None]. Include the log probabilities on the ``logprobs`` most likely tokens, as well the
            chosen tokens.
    """

    def __init__(
        self,
        model: List[str],
        prompt: Union[List[str], List[PromptSelector]],
        max_tokens: Optional[List[int]] = [None],
        stop: Optional[List[List[str]]] = [None],
        temperature: Optional[List[float]] = [None],
        top_p: Optional[List[float]] = [None],
        top_k: Optional[List[int]] = [None],
        repetition_penalty: Optional[List[float]] = [None],
        logprobs: Optional[List[int]] = [None],
    ):
        self.completion_fn = together.Complete.create

        # If we are using a prompt selector, we need to render
        # messages, as well as create prompt_keys to map the messages
        # to corresponding prompts in other models.
        if isinstance(prompt[0], PromptSelector):
            self.prompt_keys = {
                selector.for_together_completion(): selector.for_together_completion() for selector in prompt
            }
            prompt = [selector.for_together_completion() for selector in prompt]
        else:
            self.prompt_keys = prompt

        self.all_args = dict(
            model=model,
            prompt=prompt
        )

        if max_tokens == [None]:
            self.all_args["max_tokens"] = max_tokens
        if stop == [None]:
            self.all_args["stop"] = stop
        if temperature == [None]:
            self.all_args["temperature"] = temperature
        if top_p == [None]:
            self.all_args["top_p"] = top_p
        if top_k == [None]:
            self.all_args["top_k"] = top_k
        if repetition_penalty == [None]:
            self.all_args["repetition_penalty"] = repetition_penalty
        if logprobs == [None]:
            self.all_args["logprobs"] = logprobs

        super().__init__()

    @staticmethod
    def _extract_responses(output: Dict[str, object]) -> list[str]:
        return [choice["text"] for choice in output["output"]["choices"]][0]

    def _get_model_names(self):
        return [combo["model"] for combo in self.argument_combos]

    @staticmethod
    def list_models():
        return [model_dict['name'] for model_dict in together.Models.list()]