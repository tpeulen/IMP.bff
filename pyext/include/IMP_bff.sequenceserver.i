/*
 * Alignments from a remote MSA server (ColabFold's MMseqs2 API), when no local
 * sequence database is configured.
 */

%feature("compactdefaultargs") IMP::bff::fetch_server_msa;
%feature("compactdefaultargs") IMP::bff::get_a3m_hits;

%include "IMP/bff/SequenceServer.h"
