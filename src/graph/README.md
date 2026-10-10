# graph: Expression graph: nodes, ports, sessions, evaluation

The expression graph: nodes, ports, sessions, the evaluator and the node registry that builds a graph from a specification.

- **Inputs:** parameters and curves as graph values.
- **Relations:** `GraphNodeRegistry.cpp` names the fret/spectroscopy node types it constructs; nothing else here reaches upward.
- **Layout:** sources in `src/graph/`; the public headers stay flat in `include/` (IMP links only `include/*.h`) and are assigned to this theme in `src/graph/Headers.cmake`.

Public headers: `GraphEvaluation.h`, `GraphExpression.h`, `GraphNode.h`, `GraphNodeRegistry.h`, `GraphPort.h`, `GraphSession.h`.
