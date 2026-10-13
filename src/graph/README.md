# graph: Expression graph: nodes, ports, sessions, evaluation

The expression graph: nodes, ports, sessions, the evaluator and the node registry that builds a graph from a specification.

- **Inputs:** parameters and curves as graph values.
- **Relations:** `GraphNodeRegistry.cpp` names the fret/spectroscopy node types it constructs; nothing else here reaches upward.
- **Layout:** sources in `src/graph/`, public headers in `include/graph/` (reached as `<IMP/bff/graph/X.h>`; `<IMP/bff/graph.h>` includes them all). The directory is the assignment; `cmake/ThemeHeaders.cmake` links the headers into the build tree.

Public headers: `GraphEvaluation.h`, `GraphExpression.h`, `GraphNode.h`, `GraphNodeRegistry.h`, `GraphPort.h`, `GraphSession.h`.
