from typing import Callable, List, Optional, Any, Type

from cv2.typing import MatLike


class Transform:
    def __init__(
        self,
        run: Callable,
        debug: Callable = None,
        label: Optional[str] = "Pipecell",
    ):
        self._run = run
        self.label = label
        self.debug = debug
        self.parent_pipeline = None

    def run_debug(self, input):
        output = self._run(input, self.parent_pipeline)
        if not self.debug:
            debug_cb = None
        debug_cb = lambda: self.debug(output, self.parent_pipeline)
        intermediate_results = {self.label: (output, debug_cb)}
        return output, intermediate_results


class Pipeline:
    initial_input_getter = lambda: None

    def __init__(
        self,
        pipecells: List[Transform | Type["Pipeline"]],
        label: Optional[str] = "Pipeline",
        parent_pipeline=None,
        results: Optional[List[Transform]] = None,
    ):
        for cell in pipecells:
            cell.parent_pipeline = self
        self.pipecells = pipecells
        self.label = label
        self.parent_pipeline = parent_pipeline

    def _run(
        self,
        input: Any,
        parent_pipeline=None,
        results: Optional[List[Transform] | None] = None,
        aggregated_results: Optional[List[Any]] = None,
    ):
        self.initial_input_getter = lambda: input
        input_copy = input

        if results is None:
            for transform in self.pipecells:
                input_copy = transform._run(input_copy, transform.parent_pipeline)
            return input_copy

        for transform in self.pipecells:
            if isinstance(transform, Pipeline):
                input_copy = transform._run(
                    input_copy,
                    transform.parent_pipeline,
                    results=results,
                    aggregated_results=aggregated_results,
                )
            else:
                input_copy = transform._run(input_copy, transform.parent_pipeline)
            if transform in results:
                aggregated_results[results.index(transform)] = input_copy
        if self in results:
            aggregated_results[results.index(self)] = input_copy
        return input_copy

    def run(
        self,
        input: Any,
        parent_pipeline=None,
        results: Optional[List[Transform] | None] = None,
    ):
        if results is None:
            output = self._run(input, parent_pipeline)
        else:
            output = [[]] * len(results)
            self._run(input, parent_pipeline, results, output)
        return output

    def run_debug(self, input: Any):  # TODO typing
        org_input = input.copy()
        self.initial_input_getter = lambda: org_input
        intermediate_results = {}
        for pipecell in self.pipecells:
            input, cell_debug = pipecell.run_debug(input)
            intermediate_results.update(cell_debug)
        return input, intermediate_results
