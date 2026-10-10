# Public headers of the `graph` theme. IMP links only include/*.h and
# include/internal/*.h into its build tree, so headers stay flat and this
# list is what assigns them (test/test_theme_headers.py checks it).
set(imp_bff_graph_headers "GraphEvaluation.h;GraphExpression.h;GraphNode.h;GraphNodeRegistry.h;GraphPort.h;GraphSession.h")
# Sources without a public header of their own.
set(imp_bff_graph_private_sources "")
