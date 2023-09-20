from typing import Callable, List, Optional, Any, Type

from cv2.typing import MatLike

import copy


class PipelineChoice:
    def __init__(self, *choices, where):
        self.choices = choices
        self.where = where

    def run(self, input, keep_results, aggregated_results):
        output = None
        for choice in self.choices:
            input_copy = copy.deepcopy(input)
            output = Pipeline._run(input_copy, choice, keep_results, aggregated_results)
            if self.where(output):
                break
        return output


# class BindInput:


Transform = Callable | List | PipelineChoice


class Pipeline:
    initial_input_getter = lambda: None

    def __init__(
        self,
        transforms: List[Transform] | None = None,
    ):
        self.transforms = transforms
        self.initial_input = None
        self.aggregated_results = {}
        self.keep_results = []

    def choice(self, *choices, first_where):
        """Choose the first pipeline which output passes the `first_where` test function."""

        def conditionalPipeline(input):
            output = None
            for choice in choices:
                input_copy = copy.deepcopy(input)
                output = self._run(input_copy, choice)
                if first_where(output):
                    break
            return output

        return conditionalPipeline

    def bind_input(self, fn: Callable) -> Callable:
        """Bind the pipeline input as last parameter of `fn`."""
        return lambda *args, **kwargs: fn(*args, self.get_input(), **kwargs)

    def setTransforms(
        self,
        transforms: List[Transform],
    ):
        self.transforms = transforms

    def get_input(self):
        return self.initial_input

    def _run(self, input, transform):
        if isinstance(transform, Callable):
            input = transform(input)
        elif isinstance(transform, Pipeline):
            sub_pipeline = transform
            sub_pipeline.aggregated_results = self.aggregated_results
            sub_pipeline.keep_results = self.keep_results
            sub_pipeline.initial_input = input
            input = sub_pipeline._run(input, sub_pipeline.transforms)

        elif isinstance(transform, tuple):
            for subtransform in transform:
                input = self._run(
                    input,
                    subtransform,
                )
        else:
            # TODO
            print("ERROR unknow type")

        if transform in self.keep_results:
            self.aggregated_results[transform] = copy.deepcopy(input)
        return input

    def run(
        self,
        input: Any,
        keep_results: List[Transform] = [],
    ):
        self.aggregated_results = {}
        self.keep_results = keep_results
        self.initial_input = input

        if len(keep_results) is 0:
            return self._run(input, self.transforms)

        output = []
        for item in keep_results:
            self.aggregated_results[item] = []
        self._run(input, self.transforms)
        if self in keep_results:
            self.aggregated_results[self] = copy.deepcopy(input)
        for item in keep_results:
            output.append(self.aggregated_results[item])
        return output
