/**
 * \file StructureTable.cpp
 * \brief The structure record: its columns as arrays, and the residue test.
 *
 * No IMP here. Two readers fill the table from a coordinate file:
 * src/imp/StructureReader.cpp through `IMP::atom` (the connection layer's),
 * and #IMP::bff::parse_structure_table below, which follows the same rules
 * without IMP so that the core reads a structure the way the module build
 * does.
 *
 * Copyright 2007-2026 IMP Inventors. All rights reserved.
 */

#include <IMP/bff/structure/StructureTable.h>

#include <IMP/bff/structure/VdwRadii.h>
#include <IMP/bff/internal/Cif.h>
#include <IMP/bff/internal/OutputView.h>
#include <IMP/bff/internal/Text.h>
#include <IMP/bff/util/IMPCompatibility.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <set>
#include <string>
#include <vector>

IMPBFF_BEGIN_NAMESPACE

namespace {

//! The residue names IMP's own hierarchy calls standard, as strings.
/*! Written out rather than derived from `IMP::atom::ResidueType` because the
    question a caller asks is about the *name in the file*, and a name IMP has
    never seen has no ResidueType to ask. */
const std::set<std::string>& standard_residues() {
    static const std::set<std::string> s = {
            // the twenty, plus the two encoded by stop codons
            "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS",
            "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP",
            "TYR", "VAL", "SEC", "PYL",
            // protonation and tautomer spellings CHARMM and PDB both use
            "HSD", "HSE", "HSP", "HID", "HIE", "HIP", "CYX", "CYM", "ASH",
            "GLH", "LYN", "MSE",
            // nucleotides, DNA and RNA, in both the one- and two-letter
            // spellings deposited files carry
            "A", "C", "G", "T", "U", "I",
            "DA", "DC", "DG", "DT", "DU", "DI",
            "ADE", "CYT", "GUA", "THY", "URA",
            // the modified nucleotides a deposited RNA carries. They are
            // kept because dropping them cuts a tRNA into fragments, and a
            // structure with holes in its backbone is worse than one
            // carrying a residue whose chemistry this table does not model.
            "2DA", "2DC", "2DG", "2DT",
            "1MA", "1MG", "1MC", "1MT",
            "M2G", "OMG", "OMC", "H2U",
            "PSU", "5MC", "7MG"};
    return s;
}

}  // namespace

bool is_standard_residue(const std::string& res_name) {
    std::string name = res_name;
    while (!name.empty() && name.front() == ' ') name.erase(name.begin());
    while (!name.empty() && name.back() == ' ') name.erase(name.end() - 1);
    std::transform(name.begin(), name.end(), name.begin(), ::toupper);
    return standard_residues().count(name) > 0;
}

void StructureTable::get_xyz(double** out_view, int* n_out_view) const {
    double* out = internal::new_double_view(xyz.size(), out_view, n_out_view);
    if (out != nullptr && !xyz.empty()) {
        std::memcpy(out, xyz.data(), xyz.size() * sizeof(double));
    }
}

void StructureTable::get_radius(double** out_view, int* n_out_view) const {
    double* out =
            internal::new_double_view(radius.size(), out_view, n_out_view);
    if (out != nullptr && !radius.empty()) {
        std::memcpy(out, radius.data(), radius.size() * sizeof(double));
    }
}

void StructureTable::get_mass(double** out_view, int* n_out_view) const {
    double* out = internal::new_double_view(mass.size(), out_view, n_out_view);
    if (out != nullptr && !mass.empty()) {
        std::memcpy(out, mass.data(), mass.size() * sizeof(double));
    }
}

void StructureTable::get_bfactor(double** out_view, int* n_out_view) const {
    double* out =
            internal::new_double_view(bfactor.size(), out_view, n_out_view);
    if (out != nullptr && !bfactor.empty()) {
        std::memcpy(out, bfactor.data(), bfactor.size() * sizeof(double));
    }
}

void StructureTable::get_atom_id(int** out_view_i, int* n_out_view_i) const {
    int* out = internal::new_int_view(atom_id.size(), out_view_i, n_out_view_i);
    if (out != nullptr && !atom_id.empty()) {
        std::memcpy(out, atom_id.data(), atom_id.size() * sizeof(int));
    }
}

void StructureTable::get_res_id(int** out_view_i, int* n_out_view_i) const {
    int* out = internal::new_int_view(res_id.size(), out_view_i, n_out_view_i);
    if (out != nullptr && !res_id.empty()) {
        std::memcpy(out, res_id.data(), res_id.size() * sizeof(int));
    }
}


// --------------------------------------------------------------------------
// parse_structure_table: IMP's PDB and mmCIF reading rules, without IMP
// --------------------------------------------------------------------------
//
// Every rule below is one `IMP::atom::read_pdb` / `read_mmcif` applies with
// their default selector (NonWaterPDBSelector), followed by
// `internal::canonicalize`, which sorts each chain's residues by number. They
// are restated rather than linked because the core does not link IMP; the
// IMP module build holds the two readers equal on the shipped structures
// (test/representation/test_parse_structure_table_imp.py).

// A named namespace, not an anonymous one: the core is built as one unity
// translation unit, where these helper names would meet other files' own.
namespace parse_structure_detail {

//! IMP's element table (`IMP::atom::ElementTable`): symbol by atomic number.
/*! Index 0 is IMP's UNKNOWN_ELEMENT. 104 is spelled `Db` there, so an
    upper-cased lookup never finds it -- kept, since the point is IMP's
    answer, not the periodic table's. */
const char* const ELEMENT_SYMBOLS[] = {
        "Un", "H",  "HE", "LI", "BE", "B",  "C",  "N",  "O",  "F",  "NE",
        "NA", "MG", "AL", "SI", "P",  "S",  "CL", "AR", "K",  "CA", "SC",
        "TI", "V",  "CR", "MN", "FE", "CO", "NI", "CU", "ZN", "GA", "GE",
        "AS", "SE", "BR", "KR", "RB", "SR", "Y",  "ZR", "NB", "MO", "TC",
        "RU", "RH", "PD", "AG", "CD", "IN", "SN", "SB", "TE", "I",  "XE",
        "CS", "BA", "LA", "CE", "PR", "ND", "PM", "SM", "EU", "GD", "TB",
        "DY", "HO", "ER", "TM", "YB", "LU", "HF", "TA", "W",  "RE", "OS",
        "IR", "PT", "AU", "HG", "TL", "PB", "BI", "PO", "AT", "RN", "FR",
        "RA", "AC", "TH", "PA", "U",  "NP", "PU", "AM", "CM", "BK", "CF",
        "ES", "FM", "MD", "NO", "LR", "Db", "JL", "RF"};

//! IMP's element masses (`ElementTable::mass_`), Dalton, by atomic number.
const double ELEMENT_MASSES[] = {
        0.0,     1.007,   4.002,   6.941,   9.012,    10.811,  12.0107,
        14.0067, 15.9994, 18.998,  20.1797, 22.989,   24.3050, 26.981,
        28.0855, 30.973,  32.065,  35.453,  39.948,   39.0983, 40.078,
        44.955,  47.867,  50.9415, 51.9961, 54.938,   55.845,  58.933,
        58.6934, 63.546,  65.409,  69.723,  72.64,    74.921,  78.96,
        79.904,  83.798,  85.4678, 87.62,   88.905,   91.224,  92.906,
        95.94,   97.907,  101.07,  102.905, 106.42,   107.8682, 112.411,
        114.818, 118.71,  121.76,  127.60,  126.90,   131.29,  132.91,
        137.33,  138.91,  140.12,  140.91,  144.24,   145,     150.36,
        151.96,  157.25,  158.93,  162.50,  164.93,   167.26,  168.93,
        173.04,  174.97,  178.49,  180.95,  183.84,   186.21,  190.23,
        192.22,  195.08,  196.97,  200.59,  204.38,   207.2,   208.98,
        208.98,  209.99,  222.02,  223.0,   226.0,    227.0,   232.04,
        231.04,  238.03,  237.0,   244.0,   243.0,    247.0,   247.0,
        251.0,   252.0,   257.0,   258.0,   259.0,    262.0,   262.0,
        262.0,   262.0};

const int N_ELEMENTS = sizeof(ELEMENT_SYMBOLS) / sizeof(ELEMENT_SYMBOLS[0]);

//! `ElementTable::get_element`: the atomic number, 0 when IMP has no such symbol.
int element_number(const std::string& symbol) {
    static const std::map<std::string, int> numbers = [] {
        std::map<std::string, int> m;
        for (int z = 1; z < N_ELEMENTS; ++z) m[ELEMENT_SYMBOLS[z]] = z;
        return m;
    }();
    std::map<std::string, int>::const_iterator it =
            numbers.find(internal::upper(symbol));
    return it == numbers.end() ? 0 : it->second;
}

//! The element of a PDB record: `get_element_from_pdb_line` in IMP's pdb.cpp.
int pdb_element(const std::string& line, bool is_atom, bool is_hetatm) {
    if (line.size() >= 78) {
        const int z = element_number(internal::trimmed(line.substr(76, 2)));
        if (z != 0) return z;
    }
    const std::string name = line.substr(12, 4);
    if (is_atom) {
        char c0 = name[0];
        if ((std::isdigit(static_cast<unsigned char>(name[0])) ||
             std::isspace(static_cast<unsigned char>(name[0]))) &&
            std::isalpha(static_cast<unsigned char>(name[1]))) {
            c0 = name[1];
        }
        // the six most likely, so that `NE` (an arginine nitrogen) is not neon
        switch (c0) {
            case 'H': return 1;
            case 'C': return 6;
            case 'N': return 7;
            case 'O': return 8;
            case 'S': return 16;
            case 'P': return 15;
            default: break;
        }
    }
    if (is_hetatm) {
        std::string symbol;
        if (std::isalpha(static_cast<unsigned char>(name[0]))) symbol.push_back(name[0]);
        if (std::isalpha(static_cast<unsigned char>(name[1]))) symbol.push_back(name[1]);
        const int z = element_number(symbol);
        if (z != 0) return z;
    }
    return 0;
}

struct ParsedAtom {
    double x, y, z, bfactor;
    int serial, element;
    std::string name;
};

struct ParsedResidue {
    int index;
    std::string icode, name;
    std::vector<ParsedAtom> atoms;
};

struct ParsedChain {
    std::string id;
    std::vector<ParsedResidue> residues;
};

//! The single-precision value IMP keeps of a number it parsed.
inline double as_float(double v) {
    return static_cast<double>(static_cast<float>(v));
}

std::string residue_name_or_unk(const std::string& raw) {
    const std::string name = internal::trimmed(raw);
    return name.empty() ? std::string("UNK") : name;
}

//! IMP names a HETATM atom `HET:` + the raw field; the table reports the name.
std::string atom_name(const std::string& raw, bool is_hetatm) {
    const std::string name = internal::trimmed(raw);
    if (is_hetatm) return name;
    return name.empty() ? std::string("UNK") : name;
}

bool is_water(const std::string& res_name) {
    return res_name == "HOH" || res_name == "DOD";
}

std::vector<ParsedChain> parse_pdb(const std::string& path, bool keep_water) {
    std::ifstream in(path.c_str());
    if (!in) IMP_THROW("Cannot open " << path, IOException);
    std::vector<ParsedChain> chains;
    bool first_model_read = false;
    char curr_chain = 0;
    char curr_icode = 0;
    std::string line;
    while (std::getline(in, line)) {
        if (!line.empty() && line[line.size() - 1] == '\r') line.erase(line.size() - 1);
        if (line.compare(0, 5, "MODEL") == 0) {
            if (first_model_read) break;  // the first model only
            first_model_read = true;
        }
        if (line.size() < 54) continue;  // IMP's record test: through z
        const bool is_atom = line.compare(0, 4, "ATOM") == 0;
        const bool is_hetatm = line.compare(0, 6, "HETATM") == 0;
        if (!is_atom && !is_hetatm) continue;
        // NonAlternativePDBSelector, then NonWaterPDBSelector
        const char alt = line[16];
        if (alt != ' ' && alt != 'A') continue;
        const std::string raw_res = line.substr(17, 3);
        if (!keep_water && is_water(internal::trimmed(raw_res))) continue;

        // IMP reads the residue number into a `short`
        const int res_index = static_cast<short>(std::atoi(line.substr(22, 4).c_str()));
        const char icode = line[26];
        const char chain = line[21];

        if (chains.empty() || chain != curr_chain) {
            // a new run of one chain identifier is a new chain
            curr_chain = chain;
            ParsedChain c;
            c.id = std::string(1, chain);
            chains.push_back(c);
        }
        ParsedChain& c = chains.back();
        if (c.residues.empty() || res_index != c.residues.back().index ||
            icode != curr_icode) {
            curr_icode = icode;
            ParsedResidue r;
            r.index = res_index;
            r.icode = icode == ' ' ? std::string() : std::string(1, icode);
            r.name = residue_name_or_unk(raw_res);
            c.residues.push_back(r);
        }
        ParsedAtom a;
        a.x = as_float(std::atof(line.substr(30, 8).c_str()));
        a.y = as_float(std::atof(line.substr(38, 8).c_str()));
        a.z = as_float(std::atof(line.substr(46, 8).c_str()));
        a.bfactor = line.size() >= 66 ? as_float(std::atof(line.substr(60, 6).c_str())) : 0.0;
        a.serial = std::atoi(line.substr(6).c_str());
        a.element = pdb_element(line, is_atom, is_hetatm);
        a.name = atom_name(line.substr(12, 4), is_hetatm);
        c.residues.back().atoms.push_back(a);
    }
    return chains;
}

using internal::has;
using internal::txt;

//! The `_atom_site` rows, as `IMP::atom`'s AtomSiteCategory reads them.
struct AtomSiteRows {
    bool keep_water;
    ihm_keyword *group, *id, *type_symbol, *atom_id, *alt_id, *comp_id,
            *asym_id, *auth_asym_id, *seq_id, *ins_code, *auth_seq_id,
            *model_num, *x, *y, *z, *b;
    std::vector<ParsedChain> chains;
    std::map<std::string, std::size_t> chain_index;
    bool have_model;
    int model;
    long curr_chain;  // index into chains, -1 for none
    bool have_residue;
    int curr_seq_id;
    std::string curr_icode, curr_auth_seq_id, curr_label_asym;

    void row() {
        const std::string alt = txt(alt_id);
        if (!(alt.empty() || alt == " " || alt == "A")) return;
        const std::string comp = txt(comp_id);
        if (!keep_water && is_water(comp)) return;
        // the first model only: rows of another one are skipped, wherever they are
        const int m = has(model_num) ? model_num->data.ival : 0;
        if (!have_model) {
            have_model = true;
            model = m;
        } else if (m != model) {
            return;
        }

        const int seq = has(seq_id) ? seq_id->data.ival : 1;
        const std::string icode = txt(ins_code);
        const std::string label_asym = txt(asym_id);
        const std::string auth_asym = txt(auth_asym_id);
        const std::string chain = auth_asym.empty() ? label_asym : auth_asym;
        if (curr_chain < 0 || chains[curr_chain].id != chain) {
            // one chain per identifier, however its rows are spread
            std::map<std::string, std::size_t>::const_iterator it =
                    chain_index.find(chain);
            if (it == chain_index.end()) {
                ParsedChain c;
                c.id = chain;
                chains.push_back(c);
                it = chain_index.insert(std::make_pair(chain, chains.size() - 1)).first;
            }
            curr_chain = static_cast<long>(it->second);
            have_residue = false;
        }
        ParsedChain& c = chains[curr_chain];
        const std::string auth_seq = txt(auth_seq_id);
        if (!have_residue || seq != curr_seq_id || icode != curr_icode ||
            auth_seq != curr_auth_seq_id || label_asym != curr_label_asym) {
            have_residue = true;
            curr_seq_id = seq;
            curr_icode = icode;
            curr_auth_seq_id = auth_seq;
            curr_label_asym = label_asym;
            // the number and insertion code from `auth_seq_id` ("12B"), the
            // label number when it has none -- `pdbx_PDB_ins_code` only
            // separates residues, as in IMP
            const char* start = auth_seq.c_str();
            char* end = NULL;
            long number = std::strtol(start, &end, 10);
            if (end == start) number = seq;
            ParsedResidue r;
            r.index = static_cast<int>(number);
            if (end != start && *end) r.icode = std::string(1, *end);
            r.name = residue_name_or_unk(comp);
            c.residues.push_back(r);
        }
        const bool is_hetatm = txt(group) == "HETATM";
        ParsedAtom a;
        a.x = as_float(has(x) ? x->data.fval : 0.0);
        a.y = as_float(has(y) ? y->data.fval : 0.0);
        a.z = as_float(has(z) ? z->data.fval : 0.0);
        a.bfactor = as_float(has(b) ? b->data.fval : 0.0);
        a.serial = has(id) ? id->data.ival : 0;
        a.element = element_number(txt(type_symbol));
        a.name = atom_name(txt(atom_id), is_hetatm);
        c.residues.back().atoms.push_back(a);
    }

    static void callback(ihm_reader*, int, void* data, ihm_error**) {
        static_cast<AtomSiteRows*>(data)->row();
    }
};

std::vector<ParsedChain> parse_mmcif(const std::string& path, bool keep_water) {
    FILE* fh = std::fopen(path.c_str(), "r");
    if (!fh) IMP_THROW("Cannot open " << path, IOException);
    AtomSiteRows rows;
    rows.keep_water = keep_water;
    rows.have_model = false;
    rows.model = 0;
    rows.curr_chain = -1;
    rows.have_residue = false;
    rows.curr_seq_id = 0;

    ihm_file* ifile = ihm_file_new_from_fd(fileno(fh));
    ihm_reader* reader = ihm_reader_new(ifile, false);
    ihm_category* c = ihm_category_new(reader, "_atom_site", AtomSiteRows::callback,
                                       NULL, NULL, &rows, NULL);
    rows.group = ihm_keyword_str_new(c, "group_pdb");
    rows.id = ihm_keyword_int_new(c, "id");
    rows.type_symbol = ihm_keyword_str_new(c, "type_symbol");
    rows.atom_id = ihm_keyword_str_new(c, "label_atom_id");
    rows.alt_id = ihm_keyword_str_new(c, "label_alt_id");
    rows.comp_id = ihm_keyword_str_new(c, "label_comp_id");
    rows.asym_id = ihm_keyword_str_new(c, "label_asym_id");
    rows.auth_asym_id = ihm_keyword_str_new(c, "auth_asym_id");
    rows.seq_id = ihm_keyword_int_new(c, "label_seq_id");
    rows.ins_code = ihm_keyword_str_new(c, "pdbx_pdb_ins_code");
    rows.auth_seq_id = ihm_keyword_str_new(c, "auth_seq_id");
    rows.model_num = ihm_keyword_int_new(c, "pdbx_pdb_model_num");
    rows.x = ihm_keyword_float_new(c, "cartn_x");
    rows.y = ihm_keyword_float_new(c, "cartn_y");
    rows.z = ihm_keyword_float_new(c, "cartn_z");
    rows.b = ihm_keyword_float_new(c, "b_iso_or_equiv");

    bool more = false;
    ihm_error* err = NULL;
    // the first data block, as IMP reads it
    const bool ok = ihm_read_file(reader, &more, &err);
    if (!ok) {
        const std::string message = err && err->msg ? err->msg : "parse failed";
        if (err) ihm_error_free(err);
        ihm_reader_free(reader);
        std::fclose(fh);
        IMP_THROW("reading " << path << ": " << message, IOException);
    }
    ihm_reader_free(reader);
    std::fclose(fh);
    return rows.chains;
}

std::string structure_file_extension(const std::string& path) {
    const std::size_t dot = path.find_last_of('.');
    if (dot == std::string::npos) return std::string();
    std::string ext = path.substr(dot);
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    return ext;
}

bool residue_index_less(const ParsedResidue& a, const ParsedResidue& b) {
    return a.index < b.index;
}

}  // namespace parse_structure_detail

StructureTable parse_structure_table(const std::string& path, bool keep_water,
                                     bool only_standard_residues,
                                     bool radius_no_interaction) {
    using namespace parse_structure_detail;
    const std::string ext = structure_file_extension(path);
    if (ext == ".gz" || ext == ".bz2" || ext == ".zip" || ext == ".xz") {
        IMP_THROW("parse_structure_table: " << path
                  << " is compressed; decompress it first.",
                  ValueException);
    }
    const bool is_pdb = (ext == ".pdb" || ext == ".ent");
    const bool is_cif = (ext == ".cif" || ext == ".mmcif");
    if (!is_pdb && !is_cif) {
        IMP_THROW("parse_structure_table: " << path
                  << " is not a .pdb/.ent or .cif/.mmcif file",
                  ValueException);
    }
    std::vector<ParsedChain> chains =
            is_pdb ? parse_pdb(path, keep_water) : parse_mmcif(path, keep_water);

    std::size_t n = 0;
    for (std::size_t c = 0; c < chains.size(); ++c) {
        // canonicalize(): a chain's residues by number. Stable, where IMP's
        // std::sort leaves equal numbers (insertion codes) in no promised order.
        std::stable_sort(chains[c].residues.begin(), chains[c].residues.end(),
                         residue_index_less);
        for (std::size_t r = 0; r < chains[c].residues.size(); ++r) {
            n += chains[c].residues[r].atoms.size();
        }
    }
    if (n == 0) {
        IMP_THROW("No molecule read from file " << path, ValueException);
    }

    const double radius_scaling =
            radius_no_interaction ? std::pow(2.0, -1.0 / 6.0) : 1.0;
    StructureTable table;
    table.xyz.reserve(n * 3);
    for (std::size_t c = 0; c < chains.size(); ++c) {
        const ParsedChain& chain = chains[c];
        for (std::size_t r = 0; r < chain.residues.size(); ++r) {
            const ParsedResidue& residue = chain.residues[r];
            if (only_standard_residues && !is_standard_residue(residue.name)) continue;
            for (std::size_t i = 0; i < residue.atoms.size(); ++i) {
                const ParsedAtom& a = residue.atoms[i];
                table.xyz.push_back(a.x);
                table.xyz.push_back(a.y);
                table.xyz.push_back(a.z);
                // IMP's CHARMM radii are single precision
                table.radius.push_back(
                        as_float(imp_atom_radius(residue.name, a.name)) * radius_scaling);
                table.mass.push_back(ELEMENT_MASSES[a.element]);
                table.bfactor.push_back(a.bfactor);
                table.atom_id.push_back(a.serial);
                table.res_id.push_back(residue.index);
                table.chain.push_back(chain.id);
                table.res_name.push_back(residue.name);
                table.atom_name.push_back(a.name);
                table.element.push_back(ELEMENT_SYMBOLS[a.element]);
                table.insertion_code.push_back(residue.icode);
            }
        }
    }
    return table;
}

IMPBFF_END_NAMESPACE
