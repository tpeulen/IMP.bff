#ifndef IMPBFF_COMMANDLINE_H
#define IMPBFF_COMMANDLINE_H

#include <string>
#include <utility>
#include <vector>

#include "IMPCompatibility.h"

IMPBFF_BEGIN_NAMESPACE

//! The command line, compiled: one dispatcher, one group per former program.
/*!
    Every program that was a Python script in `bin/` is a group of
    subcommands of one executable, `bin/imp_bff.cpp`, whose `main` is this
    function (owner rulings 2026-09-09 and 2026-09-14). The groups carry the
    old file names minus `imp_bff_` -- `imp_bff fps ...`, `imp_bff fps-av`,
    `imp_bff labelizer`, `imp_bff traj2drot`, ... -- and the commands of the
    former `bin/imp_bff` sit at the top level. No Python program is left.

    The wheel's `imp_bff` console script calls the vector overload.

    \code
    IMP.bff.command_line_main(["traj2drot", "traj.dcd", "lib.drot.pto"])
    \endcode

    \param[in] argc,argv as `main` receives them; `argv[0]` is the program
    \return the process exit code: 0 success, 1 failure, 2 bad usage
*/
IMPBFFEXPORT int command_line_main(int argc, char** argv);

//! The words after the program name; \c args[0] names the subcommand.
IMPBFFEXPORT int command_line_main(const std::vector<std::string>& args);

//! The top-level subcommands and their one-line briefs, in dispatch order.
IMPBFFEXPORT std::vector<std::pair<std::string, std::string> > command_line_subcommands();

IMPBFF_END_NAMESPACE

#endif  // IMPBFF_COMMANDLINE_H
