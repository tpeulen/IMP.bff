/** \file ProteinSidechainRotamerLibrary.cpp
 * \brief Dunbrack side-chain tables in PTO containers.
 * Copyright 2007-2026 IMP Inventors. All rights reserved.
 */
#include <IMP/bff/structure/ProteinSidechainRotamerLibrary.h>
#include <IMP/bff/internal/OutputView.h>
#include <IMP/bff/internal/ptolib.h>
#include <IMP/bff/structure/ZMatrix.h>
#include <IMP/algebra/Vector3D.h>
#include <algorithm>
#include <limits>
#include <map>
#include <set>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <sstream>

IMPBFF_BEGIN_NAMESPACE

// ===========================================================================
// The backbone-dependent side-chain library: FASPR's binary, in the container
// ===========================================================================


namespace {

//! One record: float32 probability, then chi1..4 and sigma1..4 as int16.
const std::size_t kRecordBytes = 20;
//! 36 phi bins x 36 psi bins per residue.
const std::size_t kBinsPerResidue = 36 * 36;

//! The residues that have rotamers, in the order the binary stores them.
/*! `offset` counts (phi, psi) bin blocks from the start of the file, and
    `n_rotamers` is how many slots one bin holds -- both are FASPR's tables,
    and together they say where a residue's records begin and end. */
struct Residue {
    char code;
    int n_chi;
    int n_rotamers;
    int offset;
};

const Residue kResidues[] = {
    {'R', 4, 75,   0}, {'N', 2, 36,  75}, {'D', 2, 18, 111},
    {'C', 1,  3, 129}, {'Q', 3, 108, 132}, {'E', 3, 54, 240},
    {'H', 2, 36, 294}, {'I', 2,  9, 330}, {'L', 2,  9, 339},
    {'K', 4, 73, 348}, {'M', 3, 27, 421}, {'F', 2, 18, 448},
    {'P', 2,  2, 466}, {'S', 1,  3, 468}, {'T', 1,  3, 471},
    {'W', 2, 36, 474}, {'Y', 2, 18, 510}, {'V', 1,  3, 528},
};
const std::size_t kResidueCount = sizeof(kResidues) / sizeof(kResidues[0]);
//! What the whole library weighs: the last residue's end, in records.
const std::size_t kTotalBins = 531;   // 528 + 3, V's offset plus its rotamers

const Residue* residue(char code) {
    for (std::size_t i = 0; i < kResidueCount; ++i) {
        if (kResidues[i].code == code) return &kResidues[i];
    }
    return NULL;
}

std::size_t residue_bytes(const Residue& r) {
    return kBinsPerResidue * static_cast<std::size_t>(r.n_rotamers) *
           kRecordBytes;
}

std::size_t residue_start(const Residue& r) {
    return static_cast<std::size_t>(r.offset) * kBinsPerResidue * kRecordBytes;
}

//! FASPR's bin: 10 degrees wide, centred, wrapping at 36.
int bin_of(double angle_deg) {
    const double shifted = angle_deg + 180.0;
    int bin = static_cast<int>(std::floor((shifted + 5.0) / 10.0));
    bin %= 36;
    if (bin < 0) bin += 36;
    return bin;
}

std::string object_name(char code) {
    return std::string(1, code) + "/bbdep.records";
}

}  // namespace

void ProteinSidechainDunbrackRotamers::get_chi(double** out_view, int* n_out_view) const {
    internal::copy_to_view(chi, out_view, n_out_view);
}
void ProteinSidechainDunbrackRotamers::get_sigma(double** out_view, int* n_out_view) const {
    internal::copy_to_view(sigma, out_view, n_out_view);
}
void ProteinSidechainDunbrackRotamers::get_probability(double** out_view,
                                       int* n_out_view) const {
    internal::copy_to_view(probability, out_view, n_out_view);
}

void write_protein_sidechain_dunbrack_library(const std::string& bin_path,
                            const std::string& path) {
    if (!pto::can_compress("zstd")) {
        IMP_THROW("write_dunbrack_library: unavailable output codec 'zstd'",
                  IOException);
    }
    std::FILE* f = std::fopen(bin_path.c_str(), "rb");
    if (f == NULL) {
        IMP_THROW("write_dunbrack_library: cannot open " << bin_path,
                  IOException);
    }
    std::fseek(f, 0, SEEK_END);
    const long size = std::ftell(f);
    const std::size_t expected = kTotalBins * kBinsPerResidue * kRecordBytes;
    if (static_cast<std::size_t>(size) != expected) {
        std::fclose(f);
        IMP_THROW("write_dunbrack_library: " << bin_path << " is " << size
                  << " bytes, not the " << expected
                  << " a dun2010bbdep.bin has -- is it the FASPR library?",
                  IOException);
    }

    pto::File pto;
    if (!pto.create(path, "",
            std::string(pto::kDefaultBanner) + "\nThis container was written by IMP.bff.\nhttps://github.com/tpeulen/IMP.bff\n")) {
        IMP_THROW("PTO: cannot open " << path << " for writing: "
                  << pto.error(), IOException);
    }
    pto.set_writing_app("IMP.bff");

    std::ostringstream hj;
    hj << "{\"format\":\"rot.bbdep\",\"version\":1,\"container\":\"pto\""
       << ",\"producer\":\"IMP.bff write_dunbrack_library\""
       << ",\"source\":\"dun2010bbdep.bin\""
       << ",\"library\":\"Dunbrack 2010 backbone-dependent rotamer library\""
       << ",\"citation\":\"Shapovalov MV, Dunbrack RL. Structure "
          "2011;19:844-858\""
       << ",\"redistributed_via\":\"FASPR (Huang X, Pearce R, Zhang Y. "
          "Bioinformatics 2020;36:3758-3765), MIT\""
       << ",\"terms\":\"The rotamer library is free for academic use; this "
          "container carries it unchanged and this notice with it.\""
       << ",\"record\":{\"bytes\":20,\"fields\":[\"f32 probability\","
          "\"i16 chi1..chi4\",\"i16 sigma1..sigma4\"],"
          "\"angle_unit\":\"tenths of a degree\"}"
       << ",\"bins\":{\"phi\":36,\"psi\":36,\"width_deg\":10,"
          "\"index\":\"floor((angle + 185) / 10) mod 36\"}"
       << ",\"residues\":{";
    for (std::size_t i = 0; i < kResidueCount; ++i) {
        if (i) hj << ',';
        hj << '"' << kResidues[i].code << "\":{\"n_chi\":"
           << kResidues[i].n_chi << ",\"n_rotamers\":"
           << kResidues[i].n_rotamers << ",\"bin_offset\":"
           << kResidues[i].offset << '}';
    }
    hj << "}}";
    const std::string header = hj.str();
    std::vector<unsigned char> hb(header.begin(), header.end());
    if (pto.add_coded("rot.bbdep.header", "json", "zstd", "bbdep.json", hb.data(), hb.size(), 3) == 0) {
        IMP_THROW("PTO: writing " << pto.filename() << " failed: "
                  << pto.error(), IOException);
    }

    std::ostringstream cj;
    cj << "{\"format\":\"drot.bundle\",\"version\":10"
       << ",\"producer\":\"IMP.bff write_dunbrack_library\""
       << ",\"n_libraries\":" << kResidueCount << ",\"libraries\":[";
    for (std::size_t i = 0; i < kResidueCount; ++i) {
        if (i) cj << ',';
        cj << '"' << kResidues[i].code << '"';
    }
    cj << "]}";
    const std::string catalog = cj.str();
    std::vector<unsigned char> cb(catalog.begin(), catalog.end());
    if (pto.add_coded("drot.catalog", "json", "zstd", "drot.catalog", cb.data(), cb.size(), 3) == 0) {
        IMP_THROW("PTO: writing " << pto.filename() << " failed: "
                  << pto.error(), IOException);
    }

    for (std::size_t i = 0; i < kResidueCount; ++i) {
        const Residue& r = kResidues[i];
        const std::size_t n = residue_bytes(r);
        std::vector<unsigned char> records(n);
        if (std::fseek(f, static_cast<long>(residue_start(r)), SEEK_SET) != 0 ||
            std::fread(&records[0], 1, n, f) != n) {
            std::fclose(f);
            IMP_THROW("write_dunbrack_library: reading " << bin_path
                      << " for residue " << r.code << " failed", IOException);
        }
        if (pto.add_coded("rot.bbdep.records", "faspr20", "zstd", object_name(r.code), records.data(), records.size(), 3) == 0) {
            IMP_THROW("PTO: writing " << pto.filename() << " failed: "
                      << pto.error(), IOException);
        }
    }
    std::fclose(f);
    const bool committed = pto.commit();
    const std::string commit_error = pto.error();
    pto.close();
    if (!committed) IMP_THROW("PTO: writing " << path << " failed: "
                             << commit_error, IOException);
}

void write_protein_sidechain_dunbrack_bin(const std::string& path, const std::string& bin_path) {
    pto::File pto;
    if (!pto.open(path, false)) {
        IMP_THROW("PTO: cannot open container " << path << ": "
                  << (pto.error().empty() ? "not an EBML document with DocType pto"
                                            : pto.error()), IOException);
    }
    std::FILE* out = std::fopen(bin_path.c_str(), "wb");
    if (out == NULL) {
        IMP_THROW("write_dunbrack_bin: cannot open " << bin_path
                  << " for writing", IOException);
    }
    for (std::size_t i = 0; i < kResidueCount; ++i) {
        const Residue& r = kResidues[i];
        const auto matches = pto.find_all(object_name(r.code));
        if (matches.empty()) {
            std::fclose(out);
            IMP_THROW("write_dunbrack_bin: " << path << " has no records for "
                      << r.code, IOException);
        }
        // ptolib's reader decodes by the object's encoding.
        const std::vector<unsigned char> records = pto.read(matches.front());
        if (std::fwrite(&records[0], 1, records.size(), out) !=
            records.size()) {
            std::fclose(out);
            IMP_THROW("write_dunbrack_bin: short write to " << bin_path,
                      IOException);
        }
    }
    std::fclose(out);
}

ProteinSidechainDunbrackRotamers read_protein_sidechain_dunbrack_rotamers(const std::string& path, char residue_,
                                        double phi, double psi,
                                        double probability_min,
                                        double probability_accumulated) {
    const Residue* r = residue(residue_);
    if (r == NULL) {
        IMP_THROW("read_dunbrack_rotamers: '" << residue_
                  << "' has no backbone-dependent rotamers (alanine and "
                     "glycine have no chi; is it a one-letter code?)",
                  ValueException);
    }
    pto::File pto;
    if (!pto.open(path, false)) {
        IMP_THROW("PTO: cannot open container " << path << ": "
                  << (pto.error().empty() ? "not an EBML document with DocType pto"
                                            : pto.error()), IOException);
    }
    const auto matches = pto.find_all(object_name(r->code));
    if (matches.empty()) {
        IMP_THROW("read_dunbrack_rotamers: " << path << " has no records for "
                  << r->code, IOException);
    }
    // decoded records cover every backbone bin, regardless of codec
    const std::vector<unsigned char> records = pto.read(matches.front());
    if (records.size() != residue_bytes(*r)) {
        IMP_THROW("read_dunbrack_rotamers: " << path << " holds "
                  << records.size() << " bytes for " << r->code
                  << ", expected " << residue_bytes(*r),
                  IOException);
    }

    // Where this (phi, psi) bin's slots begin, in records.
    const std::size_t bin = static_cast<std::size_t>(36 * bin_of(phi) +
                                                     bin_of(psi));
    const std::size_t first = bin * static_cast<std::size_t>(r->n_rotamers);

    ProteinSidechainDunbrackRotamers out;
    out.n_chi = r->n_chi;
    double accumulated = 0.0;
    for (int j = 0; j < r->n_rotamers; ++j) {
        const unsigned char* rec = &records[(first + j) * kRecordBytes];
        float p;
        std::memcpy(&p, rec, sizeof(p));
        if (p < probability_min) break;
        out.probability.push_back(p);
        for (int k = 0; k < r->n_chi; ++k) {
            short chi_i, sigma_i;
            std::memcpy(&chi_i, rec + 4 + 2 * k, sizeof(chi_i));
            std::memcpy(&sigma_i, rec + 4 + 2 * (4 + k), sizeof(sigma_i));
            out.chi.push_back(chi_i / 10.0);
            out.sigma.push_back(sigma_i / 10.0);
        }
        ++out.n_rotamers;
        accumulated += p;
        if (accumulated > probability_accumulated) break;
    }
    return out;
}

// --------------------------------------------------------------------------
// Quencher rotamer ensembles
// --------------------------------------------------------------------------

void QuencherRotamerEnsemble::get_crystal(double** out_view,
                                          int* n_out_view) const {
    internal::copy_to_view(crystal, out_view, n_out_view);
}

void QuencherRotamerEnsemble::get_centres(double** out_matrix, int* n_out_rows,
                                          int* n_out_cols) const {
    int n_flat = 0;
    double* out = internal::new_double_view(centres.size(), out_matrix, &n_flat);
    *n_out_cols = 3;
    *n_out_rows = 0;
    if (out == nullptr) return;
    if (!centres.empty()) {
        std::memcpy(out, centres.data(), centres.size() * sizeof(double));
    }
    *n_out_rows = static_cast<int>(centres.size() / 3);
}

void QuencherRotamerEnsemble::get_probabilities(double** out_view,
                                                int* n_out_view) const {
    internal::copy_to_view(probabilities, out_view, n_out_view);
}

namespace qrot {

std::string key(const std::string& name) {
    std::size_t b = 0, e = name.size();
    while (b < e && std::isspace(static_cast<unsigned char>(name[b]))) ++b;
    while (e > b && std::isspace(static_cast<unsigned char>(name[e - 1]))) --e;
    std::string out = name.substr(b, e - b);
    for (std::size_t i = 0; i < out.size(); ++i) {
        out[i] = static_cast<char>(std::toupper(static_cast<unsigned char>(out[i])));
    }
    return out;
}

//! One residue's atoms, in structure order.
struct Residue {
    std::string chain, name;
    int res_id;
    std::vector<std::size_t> atoms;
};

//! A chi angle: the four atoms, and which side-chain atoms turn with it.
struct Chi {
    const char* a[4];
    //! The side-chain atoms (beyond CB) that stay put; null-terminated.
    const char* fixed[3];
};

//! chi1 then chi2, as far as this residue has them.
std::vector<Chi> chis_of(const std::string& residue) {
    std::vector<Chi> out;
    const Chi c1g = {{"N", "CA", "CB", "CG"}, {nullptr, nullptr, nullptr}};
    auto chi1 = [&](const char* g) {
        Chi c = {{"N", "CA", "CB", g}, {nullptr, nullptr, nullptr}};
        out.push_back(c);
    };
    auto chi2 = [&](const char* d) {
        // Pivot CG: CG stays, everything beyond it turns.
        Chi c = {{"CA", "CB", "CG", d}, {"CG", nullptr, nullptr}};
        out.push_back(c);
    };
    if (residue == "CYS") chi1("SG");
    else if (residue == "SER") chi1("OG");
    else if (residue == "THR") chi1("OG1");
    else if (residue == "VAL") chi1("CG1");
    else if (residue == "ILE") {
        chi1("CG1");
        Chi c = {{"CA", "CB", "CG1", "CD1"}, {"CG1", "CG2", nullptr}};
        out.push_back(c);
    } else {
        const char* d = nullptr;
        if (residue == "MET") d = "SD";
        else if (residue == "TRP" || residue == "TYR" || residue == "PHE" ||
                 residue == "LEU") d = "CD1";
        else if (residue == "HIS") d = "ND1";
        else if (residue == "ASP" || residue == "ASN") d = "OD1";
        else if (residue == "GLU" || residue == "GLN" || residue == "LYS" ||
                 residue == "ARG") d = "CD";
        if (d != nullptr) {
            out.push_back(c1g);
            chi2(d);
        }
    }
    return out;
}

char one_letter(const std::string& residue) {
    static const char* names[] = {"ALA", "ARG", "ASN", "ASP", "CYS", "GLN",
                                  "GLU", "GLY", "HIS", "ILE", "LEU", "LYS",
                                  "MET", "PHE", "PRO", "SER", "THR", "TRP",
                                  "TYR", "VAL"};
    static const char letters[] = "ARNDCQEGHILKMFPSTWYV";
    for (int i = 0; i < 20; ++i) {
        if (residue == names[i]) return letters[i];
    }
    return 0;
}

bool is_backbone(const std::string& atom) {
    return atom == "N" || atom == "CA" || atom == "C" || atom == "O";
}

algebra::Vector3D at(const std::vector<double>& xyz, std::size_t i) {
    return algebra::Vector3D(xyz[3 * i + 0], xyz[3 * i + 1], xyz[3 * i + 2]);
}

//! The residues of a structure, in order, keyed (chain, res_id, res_name).
std::vector<Residue> residues(const std::vector<std::string>& chains,
                              const std::vector<int>& res_ids,
                              const std::vector<std::string>& res_names) {
    std::vector<Residue> out;
    std::map<std::pair<std::pair<std::string, int>, std::string>, std::size_t> index;
    for (std::size_t i = 0; i < res_ids.size(); ++i) {
        const std::pair<std::pair<std::string, int>, std::string> k(
                std::make_pair(key(chains[i]), res_ids[i]), key(res_names[i]));
        std::map<std::pair<std::pair<std::string, int>, std::string>,
                 std::size_t>::const_iterator f = index.find(k);
        if (f == index.end()) {
            index[k] = out.size();
            Residue r;
            r.chain = k.first.first;
            r.res_id = k.first.second;
            r.name = k.second;
            r.atoms.push_back(i);
            out.push_back(r);
        } else {
            out[f->second].atoms.push_back(i);
        }
    }
    return out;
}

int find_atom(const Residue& r, const std::vector<std::string>& atom_names,
              const std::string& name) {
    for (std::size_t k = 0; k < r.atoms.size(); ++k) {
        if (key(atom_names[r.atoms[k]]) == name) return static_cast<int>(k);
    }
    return -1;
}

//! phi and psi of every residue (degrees), the chain-end defaults applied.
/*! Bonded when both are standard amino acids with N, CA and C, follow one
    another in the same chain, and C(i) - N(i+1) < 1.8 A -- the peptide
    builder's rule, so a gap in the deposited chain is a chain end. */
void backbone_angles(const std::vector<Residue>& rs,
                     const std::vector<std::string>& atom_names,
                     const std::vector<double>& xyz,
                     std::vector<double>& phi, std::vector<double>& psi) {
    const std::size_t n = rs.size();
    phi.assign(n, -60.0);
    psi.assign(n, -45.0);
    std::vector<int> N(n), CA(n), C(n);
    std::vector<bool> ok(n);
    for (std::size_t i = 0; i < n; ++i) {
        N[i] = find_atom(rs[i], atom_names, "N");
        CA[i] = find_atom(rs[i], atom_names, "CA");
        C[i] = find_atom(rs[i], atom_names, "C");
        ok[i] = one_letter(rs[i].name) != 0 && N[i] >= 0 && CA[i] >= 0 && C[i] >= 0;
    }
    auto pos = [&](std::size_t r, int k) { return at(xyz, rs[r].atoms[k]); };
    for (std::size_t i = 0; i + 1 < n; ++i) {
        if (!ok[i] || !ok[i + 1] || rs[i].chain != rs[i + 1].chain) continue;
        if (algebra::get_distance(pos(i, C[i]), pos(i + 1, N[i + 1])) >= 1.8) continue;
        psi[i] = dihedral_deg(pos(i, N[i]), pos(i, CA[i]), pos(i, C[i]),
                              pos(i + 1, N[i + 1]));
        phi[i + 1] = dihedral_deg(pos(i, C[i]), pos(i + 1, N[i + 1]),
                                  pos(i + 1, CA[i + 1]), pos(i + 1, C[i + 1]));
    }
}

//! Rodrigues: turn `p` about the axis through `origin` along `axis`.
algebra::Vector3D turned(const algebra::Vector3D& p,
                         const algebra::Vector3D& origin,
                         const algebra::Vector3D& axis_in, double degrees) {
    const algebra::Vector3D axis = axis_in.get_unit_vector();
    const double a = degrees * M_PI / 180.0;
    const algebra::Vector3D q = p - origin;
    const double c = std::cos(a), s = std::sin(a);
    return origin + q * c + algebra::get_vector_product(axis, q) * s +
           axis * ((q * axis) * (1.0 - c));
}

//! Set one chi of the residue's coordinates (in place) to `target` degrees.
void set_chi(std::vector<algebra::Vector3D>& xyz,
             const std::vector<std::string>& names, const Chi& chi,
             double target) {
    int idx[4];
    for (int k = 0; k < 4; ++k) {
        idx[k] = -1;
        for (std::size_t j = 0; j < names.size(); ++j) {
            if (names[j] == chi.a[k]) { idx[k] = static_cast<int>(j); break; }
        }
        if (idx[k] < 0) return;   // an incomplete side chain stays as deposited
    }
    const double current = dihedral_deg(xyz[idx[0]], xyz[idx[1]], xyz[idx[2]], xyz[idx[3]]);
    const algebra::Vector3D origin = xyz[idx[2]];
    const algebra::Vector3D axis = origin - xyz[idx[1]];
    for (std::size_t j = 0; j < names.size(); ++j) {
        const std::string& n = names[j];
        if (is_backbone(n) || n == "CB") continue;
        bool fixed = false;
        for (int f = 0; f < 3 && chi.fixed[f] != nullptr; ++f) {
            if (n == chi.fixed[f]) fixed = true;
        }
        if (fixed) continue;
        xyz[j] = turned(xyz[j], origin, axis, target - current);
    }
}

QuencherRotamerEnsemble build(const std::vector<Residue>& rs, std::size_t r,
                              const std::vector<std::string>& atom_names,
                              const std::vector<double>& coords,
                              double phi, double psi,
                              const std::string& library_path,
                              const std::vector<std::string>& quench_atoms,
                              double clash, bool* has_centre) {
    const Residue& res = rs[r];
    QuencherRotamerEnsemble out;
    out.chain = res.chain;
    out.res_id = res.res_id;
    out.res_name = res.name;
    out.phi = std::numeric_limits<double>::quiet_NaN();
    out.psi = std::numeric_limits<double>::quiet_NaN();

    std::vector<std::string> names(res.atoms.size());
    std::vector<algebra::Vector3D> xyz(res.atoms.size());
    for (std::size_t k = 0; k < res.atoms.size(); ++k) {
        names[k] = key(atom_names[res.atoms[k]]);
        xyz[k] = at(coords, res.atoms[k]);
    }
    std::set<std::string> wanted;
    for (std::size_t k = 0; k < quench_atoms.size(); ++k) wanted.insert(key(quench_atoms[k]));
    std::vector<std::size_t> centre_atoms;
    for (std::size_t k = 0; k < names.size(); ++k) {
        if (wanted.count(names[k])) centre_atoms.push_back(k);
    }
    *has_centre = !centre_atoms.empty();
    if (centre_atoms.empty()) return out;
    auto centroid = [&](const std::vector<algebra::Vector3D>& p) {
        algebra::Vector3D c(0.0, 0.0, 0.0);
        for (std::size_t k = 0; k < centre_atoms.size(); ++k) c += p[centre_atoms[k]];
        return c / static_cast<double>(centre_atoms.size());
    };
    const algebra::Vector3D crystal = centroid(xyz);
    out.crystal.assign(crystal.begin(), crystal.end());

    const std::vector<Chi> chis = chis_of(res.name);
    const char letter = one_letter(res.name);
    if (!chis.empty() && letter != 0) {
        out.phi = phi;
        out.psi = psi;
        const ProteinSidechainDunbrackRotamers rot =
                read_protein_sidechain_dunbrack_rotamers(library_path, letter, phi, psi);
        out.n_library = rot.n_rotamers;
        const int n_set = std::min<int>(rot.n_chi, static_cast<int>(chis.size()));
        // Every atom of every other residue may clash.
        std::vector<std::size_t> others;
        {
            std::vector<bool> mine(coords.size() / 3, false);
            for (std::size_t k = 0; k < res.atoms.size(); ++k) mine[res.atoms[k]] = true;
            for (std::size_t a = 0; a < mine.size(); ++a) if (!mine[a]) others.push_back(a);
        }
        std::vector<std::size_t> side;
        for (std::size_t k = 0; k < names.size(); ++k) {
            if (!is_backbone(names[k]) && names[k] != "CB") side.push_back(k);
        }
        double total = 0.0;
        for (int q = 0; q < rot.n_rotamers; ++q) {
            std::vector<algebra::Vector3D> built = xyz;
            for (int c = 0; c < n_set; ++c) {
                set_chi(built, names, chis[c], rot.chi[q * rot.n_chi + c]);
            }
            double closest = std::numeric_limits<double>::infinity();
            for (std::size_t k = 0; k < side.size(); ++k) {
                for (std::size_t o = 0; o < others.size(); ++o) {
                    closest = std::min(closest,
                                       algebra::get_distance(built[side[k]], at(coords, others[o])));
                }
            }
            if (!(closest > clash)) continue;
            const algebra::Vector3D c = centroid(built);
            out.centres.insert(out.centres.end(), c.begin(), c.end());
            out.probabilities.push_back(rot.probability[q]);
            total += rot.probability[q];
        }
        if (!out.probabilities.empty() && total > 0.0) {
            for (std::size_t k = 0; k < out.probabilities.size(); ++k) out.probabilities[k] /= total;
            out.from_library = true;
            return out;
        }
        out.centres.clear();
        out.probabilities.clear();
    }
    out.centres = out.crystal;
    out.probabilities.assign(1, 1.0);
    out.from_library = false;
    return out;
}

void check_atoms(const std::vector<std::string>& chains,
                 const std::vector<int>& res_ids,
                 const std::vector<std::string>& res_names,
                 const std::vector<std::string>& atom_names,
                 const std::vector<double>& coords) {
    const std::size_t n = res_ids.size();
    if (chains.size() != n || res_names.size() != n || atom_names.size() != n ||
        coords.size() != 3 * n) {
        IMP_THROW("one chain, residue id, residue name, atom name and three "
                  "coordinates per atom", ValueException);
    }
}

}  // namespace qrot

QuencherRotamerEnsemble quencher_rotamer_ensemble(
        const std::vector<std::string>& chains, const std::vector<int>& res_ids,
        const std::vector<std::string>& res_names,
        const std::vector<std::string>& atom_names,
        const std::vector<double>& coords, const std::string& chain,
        int res_id, const std::string& library_path,
        const std::vector<std::string>& quench_atoms, double clash) {
    qrot::check_atoms(chains, res_ids, res_names, atom_names, coords);
    const std::vector<qrot::Residue> rs = qrot::residues(chains, res_ids, res_names);
    std::vector<double> phi, psi;
    qrot::backbone_angles(rs, atom_names, coords, phi, psi);
    const std::string c = qrot::key(chain);
    for (std::size_t r = 0; r < rs.size(); ++r) {
        if (rs[r].res_id != res_id || (!c.empty() && rs[r].chain != c)) continue;
        bool has_centre = false;
        QuencherRotamerEnsemble out = qrot::build(rs, r, atom_names, coords, phi[r], psi[r],
                                                  library_path, quench_atoms, clash,
                                                  &has_centre);
        if (!has_centre) {
            IMP_THROW("residue " << chain << res_id << " has none of the "
                      "quenching atoms", ValueException);
        }
        return out;
    }
    IMP_THROW("no residue " << chain << res_id << " in the structure", ValueException);
}

QuencherRotamerEnsembles quencher_rotamer_ensembles(
        const std::vector<std::string>& chains, const std::vector<int>& res_ids,
        const std::vector<std::string>& res_names,
        const std::vector<std::string>& atom_names,
        const std::vector<double>& coords,
        const std::vector<std::string>& residue_types,
        const std::string& library_path,
        const std::map<std::string, ResidueQuenching>& table, double clash) {
    qrot::check_atoms(chains, res_ids, res_names, atom_names, coords);
    const std::map<std::string, ResidueQuenching> full =
            table.empty() ? amino_acid_quenching_defaults() : table;
    std::set<std::string> types;
    for (std::size_t t = 0; t < residue_types.size(); ++t) types.insert(qrot::key(residue_types[t]));
    const std::vector<qrot::Residue> rs = qrot::residues(chains, res_ids, res_names);
    std::vector<double> phi, psi;
    qrot::backbone_angles(rs, atom_names, coords, phi, psi);
    QuencherRotamerEnsembles out;
    for (std::size_t r = 0; r < rs.size(); ++r) {
        if (!types.count(rs[r].name)) continue;
        std::map<std::string, ResidueQuenching>::const_iterator e = full.find(rs[r].name);
        if (e == full.end() || e->second.quench_atoms.empty()) continue;
        bool has_centre = false;
        QuencherRotamerEnsemble q = qrot::build(rs, r, atom_names, coords, phi[r], psi[r],
                                                library_path, e->second.quench_atoms,
                                                clash, &has_centre);
        if (has_centre) out.push_back(q);
    }
    return out;
}

IMPBFF_END_NAMESPACE

