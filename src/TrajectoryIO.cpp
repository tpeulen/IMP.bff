/**
 * \file TrajectoryIO.cpp
 * \brief Reading rotamer-library trajectories from BinaryCIF.
 *
 * Copyright 2007-2026 IMP Inventors. All rights reserved.
 */
#include <IMP/bff/TrajectoryIO.h>
#include <IMP/bff/internal/OutputView.h>
#include <IMP/bff/internal/Text.h>
#include <IMP/bff/IMPCompatibility.h>

// The C implementation of ihm, vendored by IMP at
// modules/core/dependency/python-ihm/src/. Only the header is included: the
// symbols come from libimp_atom, which bff already links, and which exports
// them because IMP::atom::read_mmcif compiles the same reader in. Compiling
// ihm_format.c here as well would work and would duplicate it in the library.
#include "ihm_format.h"

#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <fcntl.h>
#ifdef _WIN32
#include <io.h>
#else
#include <unistd.h>
#endif
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iterator>
#include <utility>

IMPBFF_BEGIN_NAMESPACE

namespace {

//! What the per-row callback accumulates into.
struct Collector {
    std::vector<double> x, y, z;
    ihm_keyword *kx = nullptr, *ky = nullptr, *kz = nullptr;
    bool count_only = false;
    long rows = 0;
};

void on_row(ihm_reader *, int, void *data, ihm_error **) {
    Collector *c = static_cast<Collector *>(data);
    ++c->rows;
    if (c->count_only) return;
    // An omitted coordinate is a malformed trajectory rather than a missing
    // optional, so it is read as zero and left to the caller's own checks --
    // throwing from inside a C callback would cross the C boundary.
    c->x.push_back(c->kx->omitted ? 0.0 : c->kx->data.fval);
    c->y.push_back(c->ky->omitted ? 0.0 : c->ky->data.fval);
    c->z.push_back(c->kz->omitted ? 0.0 : c->kz->data.fval);
}

//! Parse \p path, filling \p c. Throws on any reader error.
void parse(const std::string& path, const std::string& category, Collector& c) {
#ifdef _WIN32
    int fd = _open(path.c_str(), O_RDONLY);
#else
    int fd = ::open(path.c_str(), O_RDONLY);
#endif
    if (fd < 0) {
        IMP_THROW("cannot open BinaryCIF trajectory " << path, IMP::IOException);
    }
    ihm_file *fh = ihm_file_new_from_fd(fd);
    ihm_reader *reader = ihm_reader_new(fh, true);        // binary
    ihm_category *cat = ihm_category_new(reader, category.c_str(), on_row,
                                         nullptr, nullptr, &c, nullptr);
    c.kx = ihm_keyword_float_new(cat, "x");
    c.ky = ihm_keyword_float_new(cat, "y");
    c.kz = ihm_keyword_float_new(cat, "z");

    bool more = false;
    ihm_error *err = nullptr;
    const bool ok = ihm_read_file(reader, &more, &err);
    std::string message;
    if (!ok) message = err && err->msg ? err->msg : "unknown error";
    ihm_reader_free(reader);
    // ihm_file_new_from_fd passes no free_func, so freeing the reader never
    // closes fd -- harmless on POSIX (unlinking an open file is fine there),
    // but a leaked Windows handle blocks the caller's own next delete or
    // reopen of this same path.
#ifdef _WIN32
    _close(fd);
#else
    close(fd);
#endif
    if (!ok) {
        IMP_THROW("reading " << path << ": " << message, IMP::IOException);
    }
}

}  // namespace

int bcif_trajectory_rows(const std::string& path, const std::string& category) {
    Collector c;
    c.count_only = true;
    parse(path, category, c);
    return static_cast<int>(c.rows);
}

void read_bcif_trajectory(const std::string& path, int n_atoms,
                          const std::string& category, double** out_view,
                          int* n_out_view) {
    if (n_atoms < 1) {
        IMP_THROW("n_atoms must be positive, not " << n_atoms,
                  IMP::ValueException);
    }
    Collector c;
    parse(path, category, c);
    const std::size_t rows = c.x.size();
    if (rows % static_cast<std::size_t>(n_atoms) != 0) {
        IMP_THROW("BinaryCIF trajectory " << path << " has " << rows
                  << " rows, which is not a multiple of n_atoms = " << n_atoms,
                  IMP::ValueException);
    }
    const std::size_t n_frames = rows / static_cast<std::size_t>(n_atoms);
    double* out = internal::new_double_view(rows * 3, out_view, n_out_view);
    if (out == nullptr) return;
    // Stored atom-major, returned frame-major: every consumer wants
    // (frame, atom, 3), and the storage order exists to make the deltas small
    // rather than to match anyone's indexing.
    for (std::size_t a = 0; a < static_cast<std::size_t>(n_atoms); ++a) {
        for (std::size_t f = 0; f < n_frames; ++f) {
            const std::size_t src = a * n_frames + f;
            const std::size_t dst = (f * n_atoms + a) * 3;
            out[dst + 0] = c.x[src];
            out[dst + 1] = c.y[src];
            out[dst + 2] = c.z[src];
        }
    }
}

// Named, not anonymous: IMP compiles this module as one translation unit.
namespace dcd {
using IMP::bff::internal::ends_with;


//! The leading Fortran record of a DCD header, in bytes.
const int HEADER_RECORD = 84;

//! Read one little- or big-endian 32-bit integer.
int read_i32(const unsigned char* p, bool big_endian) {
    int v = 0;
    if (big_endian) {
        v = (static_cast<int>(p[0]) << 24) | (static_cast<int>(p[1]) << 16) |
            (static_cast<int>(p[2]) << 8) | static_cast<int>(p[3]);
    } else {
        v = (static_cast<int>(p[3]) << 24) | (static_cast<int>(p[2]) << 16) |
            (static_cast<int>(p[1]) << 8) | static_cast<int>(p[0]);
    }
    return v;
}

float read_f32(const unsigned char* p, bool big_endian) {
    const int bits = read_i32(p, big_endian);
    float out;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}

std::vector<unsigned char> read_bytes(const std::string& path, std::size_t cap) {
    std::ifstream in(path.c_str(), std::ios::binary);
    if (!in) IMP_THROW("Cannot read " << path, IOException);
    std::vector<unsigned char> raw;
    raw.resize(cap);
    in.read(reinterpret_cast<char*>(&raw[0]), static_cast<std::streamsize>(cap));
    raw.resize(static_cast<std::size_t>(in.gcount()));
    return raw;
}

std::size_t file_size(const std::string& path) {
    std::ifstream in(path.c_str(), std::ios::binary | std::ios::ate);
    if (!in) IMP_THROW("Cannot read " << path, IOException);
    return static_cast<std::size_t>(in.tellg());
}

DCDHeader parse_header(const std::vector<unsigned char>& raw,
                       const std::string& path) {
    if (raw.size() < static_cast<std::size_t>(4 + HEADER_RECORD + 8)) {
        IMP_THROW(path << " is too short to be a DCD file", ValueException);
    }
    bool big_endian = false;
    if (read_i32(&raw[0], false) == HEADER_RECORD) {
        big_endian = false;
    } else if (read_i32(&raw[0], true) == HEADER_RECORD) {
        big_endian = true;
    } else {
        IMP_THROW("leading record is not 84 bytes in either byte order",
                  ValueException);
    }
    if (std::memcmp(&raw[4], "CORD", 4) != 0) {
        IMP_THROW(path << " does not start with the CORD magic", ValueException);
    }

    DCDHeader head;
    head.endianness = big_endian ? ">" : "<";
    head.n_frames = read_i32(&raw[8], big_endian);
    head.first_step = read_i32(&raw[12], big_endian);
    head.step_stride = read_i32(&raw[16], big_endian);
    // The float at offset 44 is the timestep; CHARMM writes it single precision.
    head.time_step = read_f32(&raw[44], big_endian);
    head.has_unit_cell = read_i32(&raw[48], big_endian) != 0;
    const bool four_dimensions = read_i32(&raw[48 + 12], big_endian) != 0;
    head.charmm_version = read_i32(&raw[84], big_endian);
    if (four_dimensions) {
        IMP_THROW(path << " is a 4-dimensional trajectory, which is not "
                          "supported",
                  ValueException);
    }

    std::size_t pos = 4 + HEADER_RECORD + 4;  // past the header record's marker
    // Title block: one Fortran record holding a count and that many 80-char
    // lines. Skipped whole; nothing here reads a title.
    const int title_len = read_i32(&raw[pos], big_endian);
    pos += 4 + static_cast<std::size_t>(title_len) + 4;

    if (pos + 8 > raw.size()) {
        IMP_THROW(path << " ends inside its atom-count record", ValueException);
    }
    const int natom_len = read_i32(&raw[pos], big_endian);
    if (natom_len != 4) {
        IMP_THROW(path << " has an unexpected atom-count record (" << natom_len
                       << " bytes)",
                  ValueException);
    }
    head.n_atoms = read_i32(&raw[pos + 4], big_endian);
    pos += 4 + static_cast<std::size_t>(natom_len) + 4;
    head.offset = static_cast<int>(pos);
    return head;
}


}  // namespace dcd

DCDHeader read_dcd_header(const std::string& path) {
    // The header, title block and atom count sit in the first few hundred
    // bytes; reading the whole trajectory to parse them made a header-only call
    // as expensive as a full load. 64 KiB is more than any title block needs.
    return dcd::parse_header(dcd::read_bytes(path, 65536), path);
}

void read_dcd(const std::string& path, int max_frames, double** out_view,
              int* n_out_view) {
    const std::vector<unsigned char> raw =
            dcd::read_bytes(path, dcd::file_size(path));
    const DCDHeader head = dcd::parse_header(raw, path);
    const bool big_endian = head.endianness == ">";

    int n_frames = head.n_frames;
    if (max_frames >= 0 && max_frames < n_frames) n_frames = max_frames;
    const std::size_t axis_bytes = 4 * static_cast<std::size_t>(head.n_atoms);

    double* out = internal::new_double_view(
            static_cast<std::size_t>(n_frames) * head.n_atoms * 3, out_view,
            n_out_view);
    if (out == nullptr) return;

    std::size_t pos = static_cast<std::size_t>(head.offset);
    for (int frame = 0; frame < n_frames; ++frame) {
        if (head.has_unit_cell) {
            // A 48-byte double record ahead of the coordinates.
            const int cell_len = dcd::read_i32(&raw[pos], big_endian);
            pos += 4 + static_cast<std::size_t>(cell_len) + 4;
        }
        for (int axis = 0; axis < 3; ++axis) {
            if (pos + 4 > raw.size()) {
                std::free(out);
                *out_view = nullptr;
                *n_out_view = 0;
                IMP_THROW(path << ": frame " << frame << " ends before its axis "
                               << axis << " record",
                          ValueException);
            }
            const std::size_t rec_len = static_cast<std::size_t>(
                    dcd::read_i32(&raw[pos], big_endian));
            if (rec_len != axis_bytes) {
                std::free(out);
                *out_view = nullptr;
                *n_out_view = 0;
                IMP_THROW(path << ": frame " << frame << " axis " << axis
                               << " record is " << rec_len << " bytes, expected "
                               << axis_bytes << " for " << head.n_atoms
                               << " atoms",
                          ValueException);
            }
            const std::size_t start = pos + 4;
            for (int i = 0; i < head.n_atoms; ++i) {
                out[(static_cast<std::size_t>(frame) * head.n_atoms + i) * 3 +
                    axis] = dcd::read_f32(&raw[start + 4 * i], big_endian);
            }
            pos = start + rec_len + 4;
        }
    }
}

void read_trajectory(const std::string& path, int n_atoms, int max_frames,
                     double** out_view, int* n_out_view) {
    if (dcd::ends_with(path, ".bcif")) {
        if (n_atoms <= 0) {
            IMP_THROW("reading a BinaryCIF trajectory needs n_atoms: it stores "
                      "one row per (atom, frame) and cannot infer where frames "
                      "divide",
                      ValueException);
        }
        double* flat = nullptr;
        int n_flat = 0;
        read_bcif_trajectory(path, n_atoms, "_rotamer_coord", &flat, &n_flat);
        std::size_t keep = static_cast<std::size_t>(n_flat);
        if (max_frames >= 0) {
            const std::size_t capped =
                    static_cast<std::size_t>(max_frames) * n_atoms * 3;
            if (capped < keep) keep = capped;
        }
        double* out = internal::new_double_view(keep, out_view, n_out_view);
        if (out != nullptr) std::memcpy(out, flat, keep * sizeof(double));
        std::free(flat);
        return;
    }
    if (dcd::ends_with(path, ".dcd")) {
        read_dcd(path, max_frames, out_view, n_out_view);
        return;
    }
    if (dcd::ends_with(path, ".xtc")) {
        const int in_file = read_xtc_n_atoms(path);
        if (n_atoms > 0 && in_file != n_atoms) {
            IMP_THROW(path << " has " << in_file << " atoms per frame, not " << n_atoms,
                      ValueException);
        }
        read_xtc(path, max_frames, out_view, n_out_view);
        return;
    }
    IMP_THROW("unsupported trajectory format: " << path, ValueException);
}

// ---------------------------------------------------------------------------
// BinaryCIF writer
// ---------------------------------------------------------------------------

// Named, not anonymous: IMP compiles this module as one translation unit.
namespace bcifw {

//! BinaryCIF ByteArray type codes (the spec's, not numpy's).
const int BYTE_ARRAY_INT8 = 1;
const int BYTE_ARRAY_INT16 = 2;
const int BYTE_ARRAY_INT32 = 3;
const int BYTE_ARRAY_FLOAT32 = 32;

//! msgpack, spelled exactly as msgpack-python's `packb(use_bin_type=True)`.
struct Packer {
    std::string out;

    void be(unsigned long long v, int bytes) {
        for (int i = bytes - 1; i >= 0; --i) {
            out.push_back(static_cast<char>((v >> (8 * i)) & 0xff));
        }
    }
    void map(std::size_t n) {
        if (n <= 15) { out.push_back(static_cast<char>(0x80 | n)); }
        else if (n <= 0xffff) { out.push_back('\xde'); be(n, 2); }
        else { out.push_back('\xdf'); be(n, 4); }
    }
    void array(std::size_t n) {
        if (n <= 15) { out.push_back(static_cast<char>(0x90 | n)); }
        else if (n <= 0xffff) { out.push_back('\xdc'); be(n, 2); }
        else { out.push_back('\xdd'); be(n, 4); }
    }
    void str(const std::string& s) {
        const std::size_t n = s.size();
        if (n <= 31) { out.push_back(static_cast<char>(0xa0 | n)); }
        else if (n < 0x100) { out.push_back('\xd9'); be(n, 1); }
        else if (n < 0x10000) { out.push_back('\xda'); be(n, 2); }
        else { out.push_back('\xdb'); be(n, 4); }
        out += s;
    }
    void bin(const std::string& b) {
        const std::size_t n = b.size();
        if (n < 0x100) { out.push_back('\xc4'); be(n, 1); }
        else if (n < 0x10000) { out.push_back('\xc5'); be(n, 2); }
        else { out.push_back('\xc6'); be(n, 4); }
        out += b;
    }
    void nil() { out.push_back('\xc0'); }
    void boolean(bool v) { out.push_back(v ? '\xc3' : '\xc2'); }
    //! msgpack-c's pack_real_int64, which msgpack-python uses for an int.
    void integer(long long d) {
        const unsigned long long u = static_cast<unsigned long long>(d);
        if (d < -(1LL << 5)) {
            if (d < -(1LL << 15)) {
                if (d < -(1LL << 31)) { out.push_back('\xd3'); be(u, 8); }
                else { out.push_back('\xd2'); be(u & 0xffffffffULL, 4); }
            } else {
                if (d < -(1LL << 7)) { out.push_back('\xd1'); be(u & 0xffffULL, 2); }
                else { out.push_back('\xd0'); be(u & 0xffULL, 1); }
            }
        } else if (d < (1LL << 7)) {
            out.push_back(static_cast<char>(u & 0xffULL));
        } else {
            if (d < (1LL << 16)) {
                if (d < (1LL << 8)) { out.push_back('\xcc'); be(u, 1); }
                else { out.push_back('\xcd'); be(u, 2); }
            } else {
                if (d < (1LL << 32)) { out.push_back('\xce'); be(u, 4); }
                else { out.push_back('\xcf'); be(u, 8); }
            }
        }
    }
};

//! Little-endian bytes of a value, as numpy's `tobytes()` writes them here.
template <typename T> void append_le(std::string& buf, T v) {
    unsigned char raw[sizeof(T)];
    std::memcpy(raw, &v, sizeof(T));
    static const int probe = 1;
    const bool host_little = *reinterpret_cast<const char*>(&probe) == 1;
    for (std::size_t i = 0; i < sizeof(T); ++i) {
        buf.push_back(static_cast<char>(raw[host_little ? i : sizeof(T) - 1 - i]));
    }
}

//! BinaryCIF IntegerPacking into int8, with escape runs.
/*! The sentinels 127 and -128 are *also* legitimate values, so a delta of
    exactly +-127 is written as an escape plus a remainder (`>=`, not `>`):
    otherwise the columns decode to different lengths. */
std::string integer_pack_int8(const std::vector<long long>& deltas) {
    const long long lo = -128, hi = 127;
    std::string out;
    for (std::size_t i = 0; i < deltas.size(); ++i) {
        long long v = deltas[i];
        while (v >= hi) { out.push_back(static_cast<char>(hi)); v -= hi; }
        while (v <= lo) { out.push_back(static_cast<char>(0x80)); v -= lo; }
        out.push_back(static_cast<char>(static_cast<unsigned long long>(v) & 0xffULL));
    }
    return out;
}

void encoding_entry(Packer& p, const char* kind, int n_extra) {
    p.map(static_cast<std::size_t>(1 + n_extra));
    p.str("kind");
    p.str(kind);
}

//! One coordinate column: FixedPoint then whichever chain is smaller.
/*! The `encoding` list is in **encode** order; `ihm_format.c` prepends each
    entry as it parses, so it decodes in the right order. */
void encode_column(Packer& p, const char* name, const std::vector<double>& values,
                   double grid_a) {
    p.map(3);
    p.str("name");
    p.str(name);
    p.str("data");
    if (grid_a <= 0.0) {
        // Lossless: float32 in, float32 out, bit-exact.
        std::string data;
        data.reserve(values.size() * 4);
        for (std::size_t i = 0; i < values.size(); ++i) {
            append_le<float>(data, static_cast<float>(values[i]));
        }
        p.map(2);
        p.str("data");
        p.bin(data);
        p.str("encoding");
        p.array(1);
        encoding_entry(p, "ByteArray", 1);
        p.str("type");
        p.integer(BYTE_ARRAY_FLOAT32);
        p.str("mask");
        p.nil();
        return;
    }
    const long long factor = static_cast<long long>(std::nearbyint(1.0 / grid_a));
    if (values.empty()) {
        IMP_THROW("column '" << name << "' is empty", ValueException);
    }
    std::vector<long long> q(values.size());
    long long span = 0;
    for (std::size_t i = 0; i < values.size(); ++i) {
        q[i] = static_cast<long long>(std::nearbyint(values[i] * static_cast<double>(factor)));
        span = (std::max)(span, q[i] < 0 ? -q[i] : q[i]);
    }
    // Chain A: plain, in the narrowest type that holds the range.
    const bool narrow = span < 32767;
    const long long plain_size = static_cast<long long>(q.size()) * (narrow ? 2 : 4);
    // Chain B: delta, then int8 packing with escape runs.
    std::vector<long long> deltas(q.size(), 0);
    long long delta_size = static_cast<long long>(q.size());
    for (std::size_t i = 1; i < q.size(); ++i) {
        deltas[i] = q[i] - q[i - 1];
        delta_size += (deltas[i] < 0 ? -deltas[i] : deltas[i]) / 127;
    }

    p.map(2);
    p.str("data");
    if (delta_size < plain_size) {
        p.bin(integer_pack_int8(deltas));
        p.str("encoding");
        p.array(4);
        encoding_entry(p, "FixedPoint", 2);
        p.str("factor"); p.integer(factor);
        p.str("srcType"); p.integer(BYTE_ARRAY_INT32);
        encoding_entry(p, "Delta", 2);
        p.str("origin"); p.integer(q[0]);
        p.str("srcType"); p.integer(BYTE_ARRAY_INT32);
        encoding_entry(p, "IntegerPacking", 3);
        p.str("byteCount"); p.integer(1);
        p.str("isUnsigned"); p.boolean(false);
        p.str("srcSize"); p.integer(static_cast<long long>(q.size()));
        encoding_entry(p, "ByteArray", 1);
        p.str("type"); p.integer(BYTE_ARRAY_INT8);
    } else {
        std::string data;
        data.reserve(q.size() * (narrow ? 2 : 4));
        for (std::size_t i = 0; i < q.size(); ++i) {
            if (narrow) append_le<int16_t>(data, static_cast<int16_t>(q[i]));
            else append_le<int32_t>(data, static_cast<int32_t>(q[i]));
        }
        p.bin(data);
        p.str("encoding");
        p.array(2);
        encoding_entry(p, "FixedPoint", 2);
        p.str("factor"); p.integer(factor);
        p.str("srcType"); p.integer(BYTE_ARRAY_INT32);
        encoding_entry(p, "ByteArray", 1);
        p.str("type"); p.integer(narrow ? BYTE_ARRAY_INT16 : BYTE_ARRAY_INT32);
    }
    p.str("mask");
    p.nil();
}

}  // namespace bcifw

long write_bcif_trajectory(const std::string& path, double* in_xyz, int n_frames,
                           int n_atoms, int n_dim, double grid_a,
                           const std::string& category, const std::string& block) {
    if (n_dim != 3 || n_frames < 0 || n_atoms < 0) {
        IMP_THROW("expected (n_frames, n_atoms, 3), got (" << n_frames << ", " << n_atoms
                  << ", " << n_dim << ")", ValueException);
    }
    const std::size_t nf = static_cast<std::size_t>(n_frames);
    const std::size_t na = static_cast<std::size_t>(n_atoms);
    bcifw::Packer p;
    p.map(3);
    p.str("version");
    p.str("0.3.0");
    p.str("encoder");
    // the Python encoder's own words, kept so the bytes are the same file
    p.str("IMP.bff scripts/trajectory_to_bcif.py");
    p.str("dataBlocks");
    p.array(1);
    p.map(2);
    p.str("header");
    p.str(block);
    p.str("categories");
    p.array(1);
    p.map(3);
    p.str("name");
    p.str(category);
    p.str("rowCount");
    p.integer(static_cast<long long>(nf * na));
    p.str("columns");
    p.array(3);
    static const char* const axes[3] = {"x", "y", "z"};
    std::vector<double> column(nf * na);
    for (int axis = 0; axis < 3; ++axis) {
        // atom-major: every frame of atom 0, then atom 1
        for (std::size_t a = 0; a < na; ++a) {
            for (std::size_t f = 0; f < nf; ++f) {
                column[a * nf + f] = in_xyz[(f * na + a) * 3 + static_cast<std::size_t>(axis)];
            }
        }
        bcifw::encode_column(p, axes[axis], column, grid_a);
    }
    std::ofstream out(path.c_str(), std::ios::binary);
    if (!out) {
        IMP_THROW("cannot write BinaryCIF trajectory " << path, IOException);
    }
    out.write(p.out.data(), static_cast<std::streamsize>(p.out.size()));
    if (!out) {
        IMP_THROW("cannot write BinaryCIF trajectory " << path, IOException);
    }
    return static_cast<long>(p.out.size());
}

// ---------------------------------------------------------------------------
// XTC reader
// ---------------------------------------------------------------------------

// Named, not anonymous: IMP compiles this module as one translation unit.
namespace xtc {

const int MAGIC = 1995;
const int FIRSTIDX = 9;
const int MAGICINTS[] = {
    0,       0,       0,       0,       0,        0,        0,        0,       0,
    8,       10,      12,      16,      20,       25,       32,       40,      50,
    64,      80,      101,     128,     161,      203,      256,      322,     406,
    512,     645,     812,     1024,    1290,     1625,     2048,     2580,    3250,
    4096,    5060,    6501,    8192,    10321,    13003,    16384,    20642,   26007,
    32768,   41285,   52015,   65536,   82570,    104031,   131072,   165140,  208063,
    262144,  330280,  416127,  524287,  660561,   832255,   1048576,  1321122, 1664510,
    2097152, 2642245, 3329021, 4194304, 5284491,  6658042,  8388607,  10568983, 13316085,
    16777216};
const int LASTIDX = static_cast<int>(sizeof(MAGICINTS) / sizeof(MAGICINTS[0]));

//! A big-endian XDR stream over a whole file.
struct Stream {
    std::vector<unsigned char> data;
    std::size_t pos = 0;
    std::string path;

    bool at_end() const { return pos >= data.size(); }
    void need(std::size_t n) const {
        if (pos + n > data.size()) {
            IMP_THROW(path << ": truncated XTC frame at byte " << pos, IOException);
        }
    }
    int i32() {
        need(4);
        const unsigned int v = (static_cast<unsigned int>(data[pos]) << 24) |
                               (static_cast<unsigned int>(data[pos + 1]) << 16) |
                               (static_cast<unsigned int>(data[pos + 2]) << 8) |
                               static_cast<unsigned int>(data[pos + 3]);
        pos += 4;
        return static_cast<int>(v);
    }
    float f32() {
        const unsigned int bits = static_cast<unsigned int>(i32());
        float v;
        std::memcpy(&v, &bits, sizeof(v));
        return v;
    }
};

//! The bit reader of xdrfile's `receivebits` / `receiveints`.
struct Bits {
    const unsigned char* buf;
    int cnt;
    unsigned int lastbits, lastbyte;
    explicit Bits(const unsigned char* b) : buf(b), cnt(0), lastbits(0), lastbyte(0) {}

    int receive(int num_of_bits) {
        const int mask = (1 << num_of_bits) - 1;
        int num = 0;
        while (num_of_bits >= 8) {
            lastbyte = (lastbyte << 8) | buf[cnt++];
            num |= static_cast<int>((lastbyte >> lastbits) << (num_of_bits - 8));
            num_of_bits -= 8;
        }
        if (num_of_bits > 0) {
            if (lastbits < static_cast<unsigned int>(num_of_bits)) {
                lastbits += 8;
                lastbyte = (lastbyte << 8) | buf[cnt++];
            }
            lastbits -= static_cast<unsigned int>(num_of_bits);
            num |= static_cast<int>((lastbyte >> lastbits) & ((1u << num_of_bits) - 1));
        }
        return num & mask;
    }

    //! Three integers packed in one mixed radix.
    void receive_ints(int num_of_bits, const unsigned int sizes[3], int nums[3]) {
        int bytes[32];
        bytes[0] = bytes[1] = bytes[2] = bytes[3] = 0;
        int num_of_bytes = 0;
        while (num_of_bits > 8) {
            bytes[num_of_bytes++] = receive(8);
            num_of_bits -= 8;
        }
        if (num_of_bits > 0) bytes[num_of_bytes++] = receive(num_of_bits);
        for (int i = 2; i > 0; --i) {
            unsigned int num = 0;
            for (int j = num_of_bytes - 1; j >= 0; --j) {
                num = (num << 8) | static_cast<unsigned int>(bytes[j]);
                const unsigned int p = num / sizes[i];
                bytes[j] = static_cast<int>(p);
                num = num - p * sizes[i];
            }
            nums[i] = static_cast<int>(num);
        }
        nums[0] = bytes[0] | (bytes[1] << 8) | (bytes[2] << 16) | (bytes[3] << 24);
    }
};

int size_of_int(unsigned int size) {
    unsigned int num = 1;
    int num_of_bits = 0;
    while (size >= num && num_of_bits < 32) {
        ++num_of_bits;
        num <<= 1;
    }
    return num_of_bits;
}

int size_of_ints(const unsigned int sizes[3]) {
    unsigned int bytes[32];
    unsigned int num_of_bytes = 1;
    bytes[0] = 1;
    int num_of_bits = 0;
    for (int i = 0; i < 3; ++i) {
        unsigned int tmp = 0;
        unsigned int bytecnt;
        for (bytecnt = 0; bytecnt < num_of_bytes; ++bytecnt) {
            tmp = bytes[bytecnt] * sizes[i] + tmp;
            bytes[bytecnt] = tmp & 0xff;
            tmp >>= 8;
        }
        while (tmp != 0) {
            bytes[bytecnt++] = tmp & 0xff;
            tmp >>= 8;
        }
        num_of_bytes = bytecnt;
    }
    unsigned int num = 1;
    --num_of_bytes;
    while (bytes[num_of_bytes] >= num) {
        ++num_of_bits;
        num *= 2;
    }
    return num_of_bits + static_cast<int>(num_of_bytes) * 8;
}

//! One frame's coordinates, float32 nanometres appended to \p nm.
void read_coords(Stream& s, int natoms, std::vector<float>& nm) {
    const int lsize = s.i32();
    if (lsize != natoms) {
        IMP_THROW(s.path << ": XTC frame says " << lsize << " atoms after a header of "
                  << natoms, IOException);
    }
    if (natoms <= 9) {
        for (int i = 0; i < 3 * natoms; ++i) nm.push_back(s.f32());
        return;
    }
    const float precision = s.f32();
    int minint[3], maxint[3];
    for (int i = 0; i < 3; ++i) minint[i] = s.i32();
    for (int i = 0; i < 3; ++i) maxint[i] = s.i32();
    unsigned int sizeint[3], bitsizeint[3] = {0, 0, 0};
    for (int i = 0; i < 3; ++i) {
        sizeint[i] = static_cast<unsigned int>(maxint[i] - minint[i] + 1);
    }
    int bitsize;
    if ((sizeint[0] | sizeint[1] | sizeint[2]) > 0xffffff) {
        for (int i = 0; i < 3; ++i) bitsizeint[i] = static_cast<unsigned int>(size_of_int(sizeint[i]));
        bitsize = 0;
    } else {
        bitsize = size_of_ints(sizeint);
    }
    int smallidx = s.i32();
    if (smallidx < FIRSTIDX || smallidx >= LASTIDX) {
        IMP_THROW(s.path << ": XTC frame has an invalid small index " << smallidx, IOException);
    }
    int smaller = MAGICINTS[(std::max)(FIRSTIDX, smallidx - 1)] / 2;
    int smallnum = MAGICINTS[smallidx] / 2;
    unsigned int sizesmall[3];
    sizesmall[0] = sizesmall[1] = sizesmall[2] = static_cast<unsigned int>(MAGICINTS[smallidx]);

    const int byte_count = s.i32();
    if (byte_count < 0) IMP_THROW(s.path << ": XTC frame has a negative size", IOException);
    const std::size_t padded =
            (static_cast<std::size_t>(byte_count) + 3) & ~static_cast<std::size_t>(3);
    s.need(padded);
    // the bit reader may look a byte past the payload; give it zeros
    std::vector<unsigned char> payload(padded + 8, 0);
    if (byte_count > 0) {
        std::memcpy(&payload[0], &s.data[s.pos], static_cast<std::size_t>(byte_count));
    }
    s.pos += padded;

    const float inv_precision = 1.0f / precision;
    Bits bits(&payload[0]);
    int thiscoord[3], prevcoord[3];
    int run = 0;
    int i = 0;
    while (i < lsize) {
        if (bitsize == 0) {
            for (int k = 0; k < 3; ++k) thiscoord[k] = bits.receive(static_cast<int>(bitsizeint[k]));
        } else {
            bits.receive_ints(bitsize, sizeint, thiscoord);
        }
        ++i;
        for (int k = 0; k < 3; ++k) {
            thiscoord[k] += minint[k];
            prevcoord[k] = thiscoord[k];
        }
        const int flag = bits.receive(1);
        int is_smaller = 0;
        if (flag == 1) {
            // `run` is kept from the last flagged atom otherwise, as xdrfile does
            run = bits.receive(5);
            is_smaller = run % 3;
            run -= is_smaller;
            --is_smaller;
        }
        if (run > 0) {
            for (int k = 0; k < run; k += 3) {
                if (i >= lsize) {
                    IMP_THROW(s.path << ": XTC run overflows the atom count", IOException);
                }
                bits.receive_ints(smallidx, sizesmall, thiscoord);
                ++i;
                for (int d = 0; d < 3; ++d) thiscoord[d] += prevcoord[d] - smallnum;
                if (k == 0) {
                    // the first two atoms of a run are swapped, for water
                    for (int d = 0; d < 3; ++d) std::swap(thiscoord[d], prevcoord[d]);
                    for (int d = 0; d < 3; ++d) nm.push_back(static_cast<float>(prevcoord[d]) * inv_precision);
                } else {
                    for (int d = 0; d < 3; ++d) prevcoord[d] = thiscoord[d];
                }
                for (int d = 0; d < 3; ++d) nm.push_back(static_cast<float>(thiscoord[d]) * inv_precision);
            }
        } else {
            for (int d = 0; d < 3; ++d) nm.push_back(static_cast<float>(thiscoord[d]) * inv_precision);
        }
        smallidx += is_smaller;
        if (smallidx < FIRSTIDX || smallidx >= LASTIDX) {
            IMP_THROW(s.path << ": XTC frame's small index left the table", IOException);
        }
        if (is_smaller < 0) {
            smallnum = smaller;
            smaller = smallidx > FIRSTIDX ? MAGICINTS[smallidx - 1] / 2 : 0;
        } else if (is_smaller > 0) {
            smaller = smallnum;
            smallnum = MAGICINTS[smallidx] / 2;
        }
        sizesmall[0] = sizesmall[1] = sizesmall[2] = static_cast<unsigned int>(MAGICINTS[smallidx]);
    }
}

Stream open_stream(const std::string& path) {
    Stream s;
    s.path = path;
    std::ifstream in(path.c_str(), std::ios::binary);
    if (!in) IMP_THROW("cannot open XTC trajectory " << path, IOException);
    s.data.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
    return s;
}

//! The frame header; returns the atom count.
int header(Stream& s) {
    const int magic = s.i32();
    if (magic != MAGIC) {
        IMP_THROW(s.path << ": not an XTC trajectory (magic " << magic << " at byte "
                  << (s.pos - 4) << ")", IOException);
    }
    const int natoms = s.i32();
    s.i32();                                    // step
    s.f32();                                    // time
    for (int k = 0; k < 9; ++k) s.f32();        // box
    return natoms;
}

}  // namespace xtc

int read_xtc_n_atoms(const std::string& path) {
    xtc::Stream s = xtc::open_stream(path);
    return xtc::header(s);
}

void read_xtc(const std::string& path, int max_frames, double** out_view, int* n_out_view) {
    xtc::Stream s = xtc::open_stream(path);
    std::vector<float> nm;
    int n_atoms = -1, frames = 0;
    while (!s.at_end() && (max_frames < 0 || frames < max_frames)) {
        const int natoms = xtc::header(s);
        if (n_atoms >= 0 && natoms != n_atoms) {
            IMP_THROW(path << ": XTC frame " << frames << " has " << natoms
                      << " atoms, the first had " << n_atoms, IOException);
        }
        n_atoms = natoms;
        xtc::read_coords(s, natoms, nm);
        ++frames;
    }
    double* out = internal::new_double_view(nm.size(), out_view, n_out_view);
    if (out == nullptr) return;
    // float32 nm as xdrfile gives it, then x10 in double, as mdtraj's caller did
    for (std::size_t i = 0; i < nm.size(); ++i) out[i] = static_cast<double>(nm[i]) * 10.0;
    if (out_view == nullptr || n_out_view == nullptr) std::free(out);
}


IMPBFF_END_NAMESPACE
