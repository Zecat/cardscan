from typing import Callable, List, Optional, Any, Type

from cv2.typing import MatLike


class Pipecell:
    def __init__(
        self,
        run: Callable,
        debug: Callable = None,
        label: Optional[str] = "Pipecell",
    ):
        self.run = run
        self.label = label
        self.debug = debug
        self.parent_pipeline = None

    def run_debug(self, input):
        output = self.run(input, self.parent_pipeline)
        if not self.debug:
            debug_cb = None
        debug_cb = lambda: self.debug(output, self.parent_pipeline)
        intermediate_results = {self.label: (output, debug_cb)}
        return output, intermediate_results


class Pipeline:
    initial_input_getter = lambda: None

    def __init__(
        self,
        pipecells: List[Pipecell | Type["Pipeline"]],
        label: Optional[str] = "Pipeline",
        parent_pipeline=None,
    ):
        for cell in pipecells:
            cell.parent_pipeline = self
        self.pipecells = pipecells
        self.label = label
        self.parent_pipeline = parent_pipeline

    def run(
        self,
        input: Any,
        parent_pipeline=None,
        results: Optional[List[str] | None] = None,
    ):
        self.initial_input_getter = lambda: input
        input_copy = input
        if results is None:
            for pipecell in self.pipecells:
                input_copy = pipecell.run(input_copy, pipecell.parent_pipeline)
            return input_copy
        else:
            intermediate_results_output = []
            for pipecell in self.pipecells:
                if isinstance(pipecell, Pipeline):
                    input_copy, intermediate_results = pipecell.run(
                        input_copy, pipecell.parent_pipeline, results=results
                    )
                    intermediate_results_output += intermediate_results
                else:
                    input_copy = pipecell.run(input_copy, pipecell.parent_pipeline)
                    if pipecell.label in results:
                        intermediate_results_output.append(input_copy)
            return input_copy, intermediate_results_output

    def run_debug(self, input: Any):  # TODO typing
        org_input = input.copy()
        self.initial_input_getter = lambda: org_input
        intermediate_results = {}
        for pipecell in self.pipecells:
            input, cell_debug = pipecell.run_debug(input)
            intermediate_results.update(cell_debug)
        return input, intermediate_results
