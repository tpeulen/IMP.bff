/**
 *  \file imp_bff.cpp
 *  \brief imp_bff -- the command line of IMP.bff, one program.
 *
 *  Every former `bin/` script is a group of subcommands here; the grammar
 *  and the groups are in the library (include/CommandLine.h), so the wheel's
 *  console script and this executable run the same code.
 *
 *  Copyright 2007-2026 IMP Inventors. All rights reserved.
 */

#include <IMP/bff/CommandLine.h>

int main(int argc, char* argv[]) {
  return IMP::bff::command_line_main(argc, argv);
}
