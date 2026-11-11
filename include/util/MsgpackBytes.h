/**
 *  \file IMP/bff/util/MsgpackBytes.h
 *  \brief The type of a msgpack document as raw bytes.
 *
 *  Moved out of learn/NeuralNet.h on 2026-11-11 (PRD-156 step 3, claude-code/Opus 5.5 for
 *  tpeulen), unchanged, because factor-graph documents use it too.
 *
 *  Copyright 2007-2026 IMP Inventors. All rights reserved.
 */
#ifndef IMPBFF_MSGPACKBYTES_H
#define IMPBFF_MSGPACKBYTES_H

#include <IMP/bff/bff_config.h>

#include <string>

IMPBFF_BEGIN_NAMESPACE

//! A msgpack document as raw bytes: bff's native format for networks and saved factor graphs.
/*! Binary-safe in C++ (a `std::string` holds any byte, including zero); in
    Python it is `bytes` both ways -- a bytes, bytearray or memoryview goes
    in, `bytes` comes out, and a `str` is refused with a TypeError. Build or
    read one with the `msgpack` package: `msgpack.packb(doc,
    use_bin_type=True)` / `msgpack.unpackb(data, raw=False)`. */
typedef std::string MsgpackBytes;

IMPBFF_END_NAMESPACE

#endif /* IMPBFF_MSGPACKBYTES_H */
