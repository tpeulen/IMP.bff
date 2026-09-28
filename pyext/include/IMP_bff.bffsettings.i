/*
 * The user's settings file: sequence databases, MSA servers, threads and
 * memory for a search.
 */

IMP_SWIG_VALUE(IMP::bff, SequenceSearchServer, SequenceSearchServers);
IMP_SWIG_VALUE(IMP::bff, SequenceSearchSettings, SequenceSearchSettingsList);

%include "IMP/bff/BffSettings.h"
