/**
 * \file IMP/bff/Registry.h
 * \brief What bff can do, by name, as data -- the same registry mechanism as tttrlib's.
 *
 * Copyright 2007-2026 IMP Inventors. All rights reserved.
 *
 * One mechanism for every library a caller reads: the descriptor, the table and the emitted entry
 * are tttrlib's `RegistryCore.h`, vendored here byte for byte (`internal/RegistryCore.h`, pinned by
 * `test/test_vendored_headers.py`), and the Python accessors `IMP.bff.registry()`, `describe()`,
 * `resolve()`, `defaults()`, `compose()` and `api_index()` are tttrlib's `registry_access.py`, embedded
 * the same way. A consumer such as ChiSurf therefore reads `tttrlib.registry()` and
 * `IMP.bff.registry()` with one piece of code; nothing is listed by hand on either side.
 *
 * Every entry is registered next to the code it describes, from a static initialiser, as in tttrlib:
 * graph node types beside their factories (`GraphNodeRegistry.cpp`), samplers beside their kernels.
 * Each entry carries at least `name`, `label`, `summary`, `description`, `params_schema` (JSON
 * Schema), `capability` and `provider` (`"imp.bff"` unless the entry says otherwise); a callable one
 * also `api` (Python paths resolved from `IMP.bff`). Categories are the capabilities, alphabetically,
 * plus `model_search`, assembled from the shipped family files (`data/model_search/index.json`) the
 * way tttrlib assembles its file-format table.
 *
 * imp.bff PRD-147 amendment A2, 2026-09-15.
 */

#ifndef IMPBFF_REGISTRY_H
#define IMPBFF_REGISTRY_H

#include <IMP/bff/bff_config.h>

#include <string>
#include <vector>

IMPBFF_BEGIN_NAMESPACE

//! Register one entry under \p capability / \p key from its JSON object.
/*!
    The entry is what `registry(capability)[key]` shows; see RegistryCore.h for the keys read from
    it. `provider` defaults to `"imp.bff"`.
    \return false when the entry is not a JSON object or \p key is taken -- a duplicate is refused,
            never overwritten, so the result cannot depend on load order.
*/
IMPBFFEXPORT bool register_algorithm_json(const std::string& capability, const std::string& key,
                                          const std::string& entry_json);

//! The whole registry, `{category: {name: entry}}`, as JSON text.
IMPBFFEXPORT std::string registry_json();

//! One category, `{name: entry}`, as JSON text; `{}` when it does not exist.
IMPBFFEXPORT std::string registry_category_json(const std::string& category);

//! The category names, in the order registry_json() emits them.
IMPBFFEXPORT std::vector<std::string> registry_categories();

//! The keys registered under \p capability, in registration order.
IMPBFFEXPORT std::vector<std::string> registry_keys(const std::string& capability);

#ifndef SWIG
//! As register_algorithm_json(), with an opaque implementation handle for the capability's own
//! dispatch (a sampler's kernel factory, say). The handle must outlive the process's use of it.
IMPBFFEXPORT bool register_algorithm_json(const std::string& capability, const std::string& key,
                                          const std::string& entry_json, void* impl);

//! The implementation handle registered under \p key, or nullptr.
IMPBFFEXPORT void* registry_impl(const std::string& key);
#endif

IMPBFF_END_NAMESPACE

#endif /* IMPBFF_REGISTRY_H */
