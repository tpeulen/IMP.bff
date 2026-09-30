/** \file CommandLineSequence.cpp
 *  \brief `imp_bff sequence-db`, `imp_bff sequence-search`, `imp_bff consurf`:
 *         sequence databases, homology search and ConSurf, all native.
 *
 *  The grammar only; the work is SequenceDatabase.h, SequenceSearch.h,
 *  SequenceHomologs.h and Consurf.h. Where a database is asked for, a `.pto`
 *  path or a name from the settings file is accepted (BffSettings.h).
 *
 *  Copyright 2007-2026 IMP Inventors. All rights reserved.
 */

#include <IMP/bff/internal/CommandLineSubs.h>

#include <cstdio>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <IMP/bff/BffSettings.h>
#include <IMP/bff/Consurf.h>
#include <IMP/bff/LabelizerFeatures.h>
#include <IMP/bff/LabelizerScore.h>
#include <IMP/bff/SequenceClusters.h>
#include <IMP/bff/SequenceDatabase.h>
#include <IMP/bff/SequenceSearch.h>

IMPBFF_BEGIN_INTERNAL_NAMESPACE
namespace cli {
namespace sequence {

struct Args {
  std::string fasta, out, name = "sequences", database, input, chain, msa, representatives,
      mapping;
  int segment_mb = 16;
  bool bytes = false;
  int threshold = SequenceSearchOptions().kmer_threshold;
  int max_candidates = SequenceSearchOptions().max_candidates;
  double evalue = SequenceSearchOptions().max_evalue;
  int threads = 0;
  int min_homologues = 0;
  bool standalone = false;
};

//! The records of a FASTA file: (name, sequence).
std::vector<std::pair<std::string, std::string> > read_fasta(const std::string& path) {
  std::ifstream in(path.c_str());
  if (!in) throw SubError("cannot read " + path);
  std::vector<std::pair<std::string, std::string> > out;
  std::string line;
  while (std::getline(in, line)) {
    if (!line.empty() && line.back() == '\r') line.pop_back();
    if (line.empty()) continue;
    if (line[0] == '>') {
      out.push_back(std::make_pair(line.substr(1), std::string()));
    } else if (!out.empty()) {
      for (char c : line)
        if (std::isalpha(static_cast<unsigned char>(c))) out.back().second.push_back(c);
    }
  }
  if (out.empty()) throw SubError(path + " holds no FASTA record");
  return out;
}

SequenceSearchOptions search_options(const Args& a) {
  SequenceSearchOptions o;
  o.kmer_threshold = a.threshold;
  o.max_candidates = a.max_candidates;
  o.max_evalue = a.evalue;
  o.threads = a.threads;
  if (o.threads == 0) o.threads = get_sequence_search_settings().threads;
  return o;
}

std::string resolve_database(const std::string& given) {
  if (!given.empty()) return given;
  const SequenceSearchSettings settings = get_sequence_search_settings();
  if (settings.default_database.empty())
    throw SubError("no --database, and the settings (" + get_settings_path() +
                   ") name no default_database");
  return settings.default_database;
}

void create(const Args& a) {
  const std::size_t n = create_sequence_database(a.fasta, a.out, a.name, a.segment_mb, !a.bytes);
  const SequenceDatabase db(a.out, a.name);
  std::cout << a.out << ": " << n << " sequences, " << db.get_number_of_residues()
            << " residues\n";
}

void pack(const Args& a) {
  const std::size_t n = repack_sequence_database(a.database, a.out, !a.bytes, a.name, a.segment_mb);
  std::cout << a.out << ": " << n << " sequences, " << (a.bytes ? "a byte a residue" : "residues Huffman-coded")
            << "\n";
}

void clustered(const Args& a) {
  const std::size_t n = create_clustered_sequence_database(a.database, a.representatives, a.out);
  std::cout << a.out << ": representatives and " << n << " members, cluster-ordered\n";
}

void cluster(const Args& a) {
  const std::size_t placed =
      create_sequence_clusters(a.database, a.representatives, a.mapping);
  const SequenceClusters c(a.representatives);
  std::cout << a.representatives << ": " << c.get_number_of_clusters() << " clusters, "
            << placed << " members placed, " << c.get_number_of_orphans() << " orphans\n";
}

//! The representatives to search through: given, or the settings' for a
//! database named there.
std::string representatives_for(const Args& a, const std::string& database) {
  if (!a.representatives.empty()) return a.representatives;
  return get_sequence_search_settings().get_representatives(database);
}

void info(const Args& a) {
  const SequenceDatabase db(resolve_database(a.database), a.name);
  std::cout << db.get_path() << "\n  residues as " << (db.get_is_packed() ? "packed (Huffman / 5 bits)" : "bytes")
            << "\n  sequences " << db.get_number_of_sequences()
            << "\n  residues  " << db.get_number_of_residues() << "\n  segments  "
            << db.get_number_of_segments() << "\n";
  if (db.get_has_members()) {
    const SequenceDatabase m = db.get_members_database();
    std::cout << "  members   " << m.get_number_of_sequences() << " sequences, "
              << m.get_number_of_residues() << " residues (searched in two stages)\n";
  }
  if (db.get_number_of_sequences() > 0) std::cout << "  first     " << db.get_header(0) << "\n";
}

void search(const Args& a) {
  const std::vector<std::pair<std::string, std::string> > records = read_fasta(a.input);
  Strings queries;
  for (const auto& r : records) queries.push_back(r.second);
  const std::string name = resolve_database(a.database);
  const SequenceDatabase db(name, a.name);
  const std::string reps = representatives_for(a, name);
  const SequenceSearchHits hits =
      db.get_has_members() ? search_clustered_sequence_database(queries, db, search_options(a),
                                                                cluster_options(a))
      : reps.empty() ? search_sequence_database(queries, db, search_options(a))
                   : search_clustered_sequence_database(queries, SequenceDatabase(reps),
                                                        SequenceClusters(reps), db,
                                                        search_options(a));
  std::ofstream file;
  if (!a.out.empty()) {
    file.open(a.out.c_str());
    if (!file) throw SubError("cannot write " + a.out);
  }
  std::ostream& out = a.out.empty() ? std::cout : file;
  // BLAST's tabular columns, 1-based inclusive ranges
  out << "# query\ttarget\tidentity\tquery_start\tquery_end\ttarget_start\ttarget_end\t"
         "evalue\tbits\tscore\n";
  char buf[512];
  for (const SequenceSearchHit& h : hits) {
    const std::string query = records[static_cast<std::size_t>(h.query)].first;
    std::snprintf(buf, sizeof buf, "\t%.3f\t%d\t%d\t%d\t%d\t%.3e\t%.1f\t%d\n", h.identity,
                  h.query_start + 1, h.query_end, h.target_start + 1, h.target_end, h.evalue,
                  h.bits, h.score);
    out << query.substr(0, query.find_first_of(" \t")) << "\t" << h.identifier << buf;
  }
  if (!a.msa.empty()) {
    if (records.size() != 1) throw SubError("--msa needs exactly one query");
    std::ofstream m(a.msa.c_str());
    if (!m) throw SubError("cannot write " + a.msa);
    m << ">" << records[0].first << "\n" << records[0].second << "\n";
    for (const SequenceSearchHit& h : hits) m << ">" << h.identifier << "\n" << h.aligned << "\n";
  }
}

void consurf(const Args& a) {
  std::string query;
  Strings labels;
  const std::string suffix = path_suffix(a.input);
  const bool fasta = suffix == ".fasta" || suffix == ".fa" || suffix == ".fas" ||
                     suffix == ".faa" || suffix == ".seq";
  if (fasta) {
    query = read_fasta(a.input)[0].second;
  } else {
    // a structure: the chain's residues are the query, and label the rows
    const LabelizerStructure s = labelizer_read_structure(a.input);
    std::string chain = a.chain;
    for (const LabelizerResidue& r : s.residues) {
      if (chain.empty()) chain = r.chain;
      if (r.chain != chain) continue;
      query.push_back(labelizer_one_letter(r.comp_id));
      labels.push_back(r.comp_id + std::to_string(r.seq_id) + ":" + r.chain);
    }
    if (query.empty()) throw SubError("no residues of chain " + chain + " in " + a.input);
  }
  ConsurfOptions options;
  options.database = resolve_database(a.database);
  options.representatives = a.representatives;
  options.clusters = cluster_options(a);
  options.search = search_options(a);
  if (a.standalone) options.homologs = SequenceHomologOptions::consurf_standalone();
  const ConsurfResult r = compute_consurf(Strings(1, query), options)[0];
  if (!r.get_is_ok()) throw SubError(r.status);
  write_consurf_grades(r, a.out, labels);
  if (!a.msa.empty()) write_consurf_msa(r, a.msa);
  std::cout << a.out << ": " << r.conservation.get_n_positions() << " positions, "
            << r.homologs.size() << " homologues\n";
}

SequenceClusterSearchOptions cluster_options(const Args& a) {
  SequenceClusterSearchOptions o;
  o.min_homologues = a.min_homologues;
  return o;
}

void add_search_options(CLI::App* sub, Args& a) {
  sub->add_option("--min-homologues", a.min_homologues,
                  "clustered databases: stop the second stage once every query has this many "
                  "homologues (E <= 1e-4, >= 35 % identity), best clusters first; 0 searches all");
  sub->add_option("--representatives", a.representatives,
                  "cluster representatives: search in two stages (default: the settings')");
  sub->add_option("--threshold", a.threshold,
                  "k-mer neighbourhood score; lower is more sensitive and slower");
  sub->add_option("--max-candidates", a.max_candidates, "targets aligned per query");
  sub->add_option("--evalue", a.evalue, "report hits with at most this E-value");
  sub->add_option("--threads", a.threads, "threads; 0: the settings', else every core");
}

}  // namespace sequence

void add_sequence_subs(CLI::App& app) {
  std::shared_ptr<sequence::Args> a = std::make_shared<sequence::Args>();

  CLI::App* db = app.add_subcommand("sequence-db", R"doc(Build and inspect sequence databases.)doc");
  db->require_subcommand(1);
  CLI::App* create = db->add_subcommand("create", R"doc(FASTA or FASTA.gz into a sequence database (.pto).)doc");
  create->footer(R"doc(Streams the FASTA into binary FASTA inside a .pto container: one byte per
residue, read in place later; headers compressed. Memory is bounded whatever the
size (UniRef90 converts in a few hundred MB). Name the result in the settings
file to search it by name:

    imp_bff sequence-db create uniref90.fasta.gz /data/uniref90.pto)doc");
  create->add_option("fasta", a->fasta, "the FASTA (.gz needs zlib at build time)")
      ->required()
      ->check(CLI::ExistingFile);
  create->add_option("out", a->out, "the .pto to write")->required();
  create->add_option("--name", a->name, "the store inside the container");
  create->add_option("--segment-mb", a->segment_mb, "segment size in MB");
  create->add_flag("--bytes", a->bytes, "a byte per residue instead of packed (read in place)");
  create->callback([a] {
    set_current_sub("sequence-db create");
    sequence::create(*a);
  });
  CLI::App* pack = db->add_subcommand(
      "pack", R"doc(Rewrite a database with residues packed: Huffman-coded (or --bytes).)doc");
  pack->footer(R"doc(One streaming pass; rows keep their order, so a cluster membership built
against the old file still holds:

    imp_bff sequence-db pack uniref90.pto uniref90.packed.pto)doc");
  pack->add_option("database", a->database, "the database to rewrite")->required();
  pack->add_option("out", a->out, "the new .pto")->required();
  pack->add_flag("--bytes", a->bytes, "a byte per residue instead");
  pack->callback([a] {
    set_current_sub("sequence-db pack");
    sequence::pack(*a);
  });
  CLI::App* cluster = db->add_subcommand(
      "cluster", R"doc(Record which members belong to each cluster representative.)doc");
  cluster->footer(R"doc(A search then reads the representatives first and only the members of the
clusters found. For UniRef, with UniProt's ID mapping (UniRef90 in column 9,
UniRef50 in column 10):

    imp_bff sequence-db cluster uniref90.pto uniref50.pto idmapping_selected.tab.gz

and in the settings: "clusters": {"uniref90": ".../uniref50.pto"}.)doc");
  cluster->add_option("members", a->database, "the member database")->required();
  cluster->add_option("representatives", a->representatives,
                      "the representatives' database; the membership is added to it")
      ->required();
  cluster->add_option("mapping", a->mapping, "the mapping table (.gz with zlib)")
      ->required()
      ->check(CLI::ExistingFile);
  cluster->callback([a] {
    set_current_sub("sequence-db cluster");
    sequence::cluster(*a);
  });
  CLI::App* clustered = db->add_subcommand(
      "clustered", R"doc(One cluster-ordered database: representatives, each followed by its members.)doc");
  clustered->footer(R"doc(From a member database and its representatives with their membership
(sequence-db cluster). Searches of the result run in two stages by themselves,
reading each hit cluster's members as one run:

    imp_bff sequence-db clustered uniref90.pto uniref50.pto uniref.pto)doc");
  clustered->add_option("members", a->database, "the member database")->required();
  clustered->add_option("representatives", a->representatives,
                        "the representatives, with their membership")->required();
  clustered->add_option("out", a->out, "the clustered .pto")->required();
  clustered->callback([a] {
    set_current_sub("sequence-db clustered");
    sequence::clustered(*a);
  });
  CLI::App* info = db->add_subcommand("info", R"doc(Summarise a sequence database.)doc");
  info->add_option("database", a->database, "a .pto, or a database name from the settings");
  info->callback([a] {
    set_current_sub("sequence-db info");
    sequence::info(*a);
  });

  CLI::App* search = app.add_subcommand(
      "sequence-search", R"doc(Search a sequence database: MMseqs2's method, native.)doc");
  search->footer(R"doc(Every query of the FASTA in one pass over the database; hits as a tab table
(BLAST's columns). --msa writes the query-anchored alignment of a single query:

    imp_bff sequence-search query.fasta --database uniref90 -o hits.tsv --msa hits.fasta)doc");
  search->add_option("query", a->input, "a FASTA of queries")->required()->check(CLI::ExistingFile);
  search->add_option("--database", a->database, "a .pto, or a name from the settings");
  search->add_option("-o,--output", a->out, "the hit table; stdout by default");
  search->add_option("--msa", a->msa, "the query-anchored alignment (one query)");
  sequence::add_search_options(search, *a);
  search->callback([a] {
    set_current_sub("sequence-search");
    sequence::search(*a);
  });

  CLI::App* consurf = app.add_subcommand(
      "consurf", R"doc(ConSurf's conservation grades of a chain, computed here.)doc");
  consurf->footer(R"doc(Search, ConSurf's choice of homologues, the alignment, Rate4Site's rates and
the nine grades -- no external program and no server. The input is a structure
(one chain's residues, labelled in the output) or a FASTA sequence; the output is
ConSurf's .grades layout, which `imp_bff labelizer --conservation` reads:

    imp_bff consurf 1lk2.pdb --chain A --database uniref90 -o 1lk2_A.grades
    imp_bff labelizer 1lk2.pdb --conservation 1lk2_A.grades

--standalone applies stand-alone ConSurf's homologue rules (PSI-BLAST on
Swiss-Prot: E <= 1e-3, no identity floor, the best 149) instead of the server's.)doc");
  consurf->add_option("input", a->input, "a structure (PDB/mmCIF) or a FASTA")
      ->required()
      ->check(CLI::ExistingFile);
  consurf->add_option("--chain", a->chain, "the chain of a structure; the first by default");
  consurf->add_option("--database", a->database, "a .pto, or a name from the settings");
  consurf->add_option("-o,--output", a->out, "the .grades file")->required();
  consurf->add_option("--msa", a->msa, "also write the alignment (aligned FASTA)");
  consurf->add_flag("--standalone", a->standalone, "stand-alone ConSurf's homologue rules");
  sequence::add_search_options(consurf, *a);
  consurf->callback([a] {
    set_current_sub("consurf");
    sequence::consurf(*a);
  });
}

}  // namespace cli
IMPBFF_END_INTERNAL_NAMESPACE
