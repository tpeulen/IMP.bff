/**
 *  \file IMP/bff/BffSettings.h
 *  \brief The user's settings file: where things are on this machine.
 *
 * What is not a property of the method but of the installation -- where the
 * sequence databases live, which MSA server to fall back to, how many threads
 * and how much memory a search may take -- is read from one JSON file rather
 * than compiled in. The file is `$IMP_BFF_SETTINGS` when set, else
 * `~/.config/imp.bff/settings.json` (`%APPDATA%\imp.bff\settings.json` on
 * Windows). A missing file is an empty configuration, not an error.
 *
 * \code{.json}
 * {
 *   "sequence_search": {
 *     "databases": {"uniref90": "/Volumes/SD1TB/sequences/uniref90.pto"},
 *     "default_database": "uniref90",
 *     "indexes": {"uniref90": "/Volumes/SD1TB/sequences/uniref90.index.pto"},
 *     "servers": {
 *       "colabfold": {"url": "https://api.colabfold.com",
 *                     "protocol": "colabfold-v1",
 *                     "poll_seconds": 5, "timeout_seconds": 3600}
 *     },
 *     "fallback_server": "colabfold",
 *     "threads": 0,
 *     "memory_budget_mb": 2048
 *   }
 * }
 * \endcode
 *
 * A key in a known section that nothing reads is refused: in a settings file
 * that is nearly always a misspelling, and a misspelt database path silently
 * ignored becomes a search of nothing.
 *
 * \authors Thomas-Otavio Peulen
 *  Copyright 2007-2026 IMP Inventors. All rights reserved.
 */
#ifndef IMPBFF_BFFSETTINGS_H
#define IMPBFF_BFFSETTINGS_H

#include <IMP/bff/bff_config.h>
#include <IMP/bff/IMPCompatibility.h>

#include <map>
#include <string>
#include <vector>

IMPBFF_BEGIN_NAMESPACE

//! The settings file this process reads: `$IMP_BFF_SETTINGS`, else the
//! per-user default. The file need not exist.
IMPBFFEXPORT std::string get_settings_path();

//! A remote MSA service, as the settings describe it.
class IMPBFFEXPORT SequenceSearchServer {
public:
    std::string name;
    std::string url;
    //! How to talk to it; `"colabfold-v1"` is ColabFold's MMseqs2 API.
    std::string protocol = "colabfold-v1";
    double poll_seconds = 5.0;
    double timeout_seconds = 3600.0;

    IMP_SHOWABLE_INLINE(SequenceSearchServer,
                        out << "SequenceSearchServer(" << name << ", " << url << ", "
                            << protocol << ")");
};
IMP_VALUES(SequenceSearchServer, SequenceSearchServers);

//! The `sequence_search` section of the settings.
class IMPBFFEXPORT SequenceSearchSettings {
public:
    //! The database searched when none is named.
    std::string default_database;
    //! The server asked when no local database is configured; empty for none.
    std::string fallback_server;
    //! Search threads; 0 uses every core.
    int threads = 0;
    //! What a search may hold in memory, in MB.
    double memory_budget_mb = 2048.0;

    std::vector<std::string> get_database_names() const;
    bool get_has_database(const std::string& name) const;
    //! The `.pto` path of database \p name. \throw ValueException if unknown.
    std::string get_database(const std::string& name) const;
    void set_database(const std::string& name, const std::string& path);

    //! The k-mer index of database \p name, or "" when it has none.
    std::string get_index(const std::string& name) const;
    void set_index(const std::string& name, const std::string& path);

    std::vector<std::string> get_server_names() const;
    bool get_has_server(const std::string& name) const;
    //! \throw ValueException if unknown.
    SequenceSearchServer get_server(const std::string& name) const;
    void set_server(const SequenceSearchServer& server);

    IMP_SHOWABLE_INLINE(SequenceSearchSettings,
                        out << "SequenceSearchSettings(" << databases_.size()
                            << " databases, " << servers_.size() << " servers)");

private:
    std::map<std::string, std::string> databases_, indexes_;
    std::map<std::string, SequenceSearchServer> servers_;
};
IMP_VALUES(SequenceSearchSettings, SequenceSearchSettingsList);

//! The `sequence_search` section of the settings file at \p path, or of
//! #get_settings_path when \p path is empty.
/*! A missing file or section gives the defaults. Relative database and index
    paths are taken relative to the settings file.
    \throw ValueException on invalid JSON or an unknown key in the section */
IMPBFFEXPORT SequenceSearchSettings
get_sequence_search_settings(const std::string& path = "");

IMPBFF_END_NAMESPACE

#endif /* IMPBFF_BFFSETTINGS_H */
