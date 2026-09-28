/**
 * \file SequenceServer.cpp
 * \brief Alignments from a remote MSA server (ColabFold's MMseqs2 API).
 *
 * Copyright 2007-2026 IMP Inventors. All rights reserved.
 */
#include <IMP/bff/SequenceServer.h>
#include <IMP/bff/internal/json.h>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <thread>
#include <vector>

#ifdef IMP_BFF_HAS_ZLIB
#include <zlib.h>
#endif
#ifdef IMP_BFF_HAS_CURL
#include <curl/curl.h>
#endif

IMPBFF_BEGIN_NAMESPACE

namespace {

#ifdef IMP_BFF_HAS_ZLIB
//! A gzip stream, decompressed in memory.
std::string server_gunzip(const std::string& in) {
    z_stream zs;
    std::memset(&zs, 0, sizeof zs);
    if (inflateInit2(&zs, 16 + MAX_WBITS) != Z_OK)
        IMP_THROW("MSA server: cannot start gzip decompression", IOException);
    zs.next_in = reinterpret_cast<Bytef*>(const_cast<char*>(in.data()));
    zs.avail_in = static_cast<uInt>(in.size());
    std::string out;
    char chunk[1 << 16];
    int status = Z_OK;
    while (status == Z_OK) {
        zs.next_out = reinterpret_cast<Bytef*>(chunk);
        zs.avail_out = sizeof chunk;
        status = inflate(&zs, Z_NO_FLUSH);
        if (status != Z_OK && status != Z_STREAM_END) {
            inflateEnd(&zs);
            IMP_THROW("MSA server: the result is not valid gzip", IOException);
        }
        out.append(chunk, sizeof chunk - zs.avail_out);
    }
    inflateEnd(&zs);
    return out;
}
#endif

//! One file of a tar archive in memory, by its name without directories;
//! empty when absent. POSIX ustar: 512-byte headers, octal sizes.
std::string server_untar(const std::string& tar, const std::string& wanted) {
    std::size_t at = 0;
    while (at + 512 <= tar.size()) {
        const char* h = tar.data() + at;
        if (h[0] == '\0') break;   // the two zero blocks at the end
        const std::string name(h, strnlen(h, 100));
        const std::string prefix(h + 345, strnlen(h + 345, 155));
        const std::string full = prefix.empty() ? name : prefix + "/" + name;
        const std::size_t size = static_cast<std::size_t>(std::strtoull(std::string(h + 124, 12).c_str(), nullptr, 8));
        const char type = h[156];
        const std::size_t slash = full.find_last_of('/');
        const std::string base = slash == std::string::npos ? full : full.substr(slash + 1);
        at += 512;
        if ((type == '0' || type == '\0') && base == wanted) {
            if (at + size > tar.size()) IMP_THROW("MSA server: truncated archive", IOException);
            return tar.substr(at, size);
        }
        at += (size + 511) / 512 * 512;
    }
    return std::string();
}

#ifdef IMP_BFF_HAS_CURL
size_t server_collect(char* data, size_t size, size_t n, void* into) {
    static_cast<std::string*>(into)->append(data, size * n);
    return size * n;
}

//! One HTTP request: GET, or POST of a form when \p form is given.
std::string server_http(const std::string& url, const std::vector<std::pair<std::string, std::string> >* form,
                 double timeout_seconds) {
    static const bool initialised = curl_global_init(CURL_GLOBAL_DEFAULT) == 0;
    (void)initialised;
    CURL* h = curl_easy_init();
    if (h == nullptr) IMP_THROW("MSA server: cannot start libcurl", IOException);
    std::string body, post;
    if (form != nullptr) {
        for (const auto& kv : *form) {
            char* value = curl_easy_escape(h, kv.second.c_str(), static_cast<int>(kv.second.size()));
            if (!post.empty()) post += "&";
            post += kv.first + "=" + value;
            curl_free(value);
        }
        curl_easy_setopt(h, CURLOPT_POSTFIELDS, post.c_str());
    }
    curl_easy_setopt(h, CURLOPT_URL, url.c_str());
    curl_easy_setopt(h, CURLOPT_WRITEFUNCTION, server_collect);
    curl_easy_setopt(h, CURLOPT_WRITEDATA, &body);
    curl_easy_setopt(h, CURLOPT_USERAGENT, "imp.bff");
    curl_easy_setopt(h, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(h, CURLOPT_TIMEOUT_MS, static_cast<long>(timeout_seconds * 1000));
    const CURLcode rc = curl_easy_perform(h);
    long code = 0;
    curl_easy_getinfo(h, CURLINFO_RESPONSE_CODE, &code);
    curl_easy_cleanup(h);
    if (rc != CURLE_OK)
        IMP_THROW("MSA server: " << url << ": " << curl_easy_strerror(rc), IOException);
    if (code >= 400) IMP_THROW("MSA server: " << url << ": HTTP " << code, IOException);
    return body;
}

nlohmann::json server_ticket(const std::string& text, const std::string& url) {
    try {
        return nlohmann::json::parse(text);
    } catch (const std::exception&) {
        IMP_THROW("MSA server: " << url << " did not answer with JSON", IOException);
    }
}
#endif

std::string server_trim_slash(std::string url) {
    while (!url.empty() && url.back() == '/') url.pop_back();
    return url;
}

}  // namespace

bool get_sequence_server_available() {
#if defined(IMP_BFF_HAS_CURL) && defined(IMP_BFF_HAS_ZLIB)
    return true;
#else
    return false;
#endif
}

std::string fetch_server_msa(const std::string& query, const SequenceSearchServer& server,
                             const std::string& mode) {
    if (server.protocol != "colabfold-v1")
        IMP_THROW("fetch_server_msa: unknown protocol '" << server.protocol
                  << "' of server " << server.name << " (known: colabfold-v1)", ValueException);
#if defined(IMP_BFF_HAS_CURL) && defined(IMP_BFF_HAS_ZLIB)
    const std::string base = server_trim_slash(server.url);
    const double request_timeout = std::min(server.timeout_seconds, 600.0);
    std::string sequence;
    for (char c : query)
        if (std::isalpha(static_cast<unsigned char>(c)))
            sequence.push_back(static_cast<char>(std::toupper(static_cast<unsigned char>(c))));
    const std::vector<std::pair<std::string, std::string> > form = {
            {"q", ">101\n" + sequence + "\n"}, {"mode", mode}};
    nlohmann::json state = server_ticket(server_http(base + "/ticket/msa", &form, request_timeout),
                                  base + "/ticket/msa");
    const auto started = std::chrono::steady_clock::now();
    auto status = [&]() { return state.value("status", std::string("UNKNOWN")); };
    while (status() == "UNKNOWN" || status() == "RUNNING" || status() == "PENDING") {
        const double waited = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
        if (waited > server.timeout_seconds)
            IMP_THROW("fetch_server_msa: " << server.name << " did not finish within "
                      << server.timeout_seconds << " s", IOException);
        std::this_thread::sleep_for(std::chrono::duration<double>(server.poll_seconds));
        const std::string id = state.value("id", std::string());
        if (id.empty()) IMP_THROW("fetch_server_msa: " << server.name << " gave no ticket", IOException);
        state = server_ticket(server_http(base + "/ticket/" + id, nullptr, request_timeout), base + "/ticket/" + id);
    }
    if (status() != "COMPLETE")
        IMP_THROW("fetch_server_msa: " << server.name << " answered " << status(), IOException);
    const std::string archive =
            server_http(base + "/result/download/" + state.value("id", std::string()), nullptr, request_timeout);
    const std::string a3m = server_untar(server_gunzip(archive), "uniref.a3m");
    if (a3m.empty()) IMP_THROW("fetch_server_msa: the result holds no uniref.a3m", IOException);
    return a3m;
#else
    (void)query;
    (void)mode;
    IMP_THROW("fetch_server_msa: this imp.bff was built without libcurl and zlib; "
              "configure a local sequence database instead", IOException);
#endif
}

SequenceSearchHits get_a3m_hits(const std::string& a3m, int query_index) {
    // records: header, sequence (lines joined)
    std::vector<std::pair<std::string, std::string> > records;
    std::istringstream in(a3m);
    std::string line;
    while (std::getline(in, line)) {
        line.erase(std::remove(line.begin(), line.end(), '\0'), line.end());
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty() || line[0] == '#') continue;
        if (line[0] == '>') {
            records.push_back(std::make_pair(line.substr(1), std::string()));
        } else if (!records.empty()) {
            for (char c : line)
                if (!std::isspace(static_cast<unsigned char>(c))) records.back().second.push_back(c);
        }
    }
    if (records.empty()) IMP_THROW("get_a3m_hits: no sequences", ValueException);
    std::string query;
    for (char c : records[0].second)
        if (std::isupper(static_cast<unsigned char>(c)) || c == '-') query.push_back(c);
    const int n = static_cast<int>(query.size());
    SequenceSearchHits out;
    for (std::size_t r = 1; r < records.size(); ++r) {
        const std::string& row = records[r].second;
        SequenceSearchHit h;
        h.query = query_index;
        h.query_length = n;
        h.aligned.assign(static_cast<std::size_t>(n), '-');
        // first pass: match columns
        int column = 0, first = -1, last = -1, residues = 0;
        for (char c : row) {
            if (std::islower(static_cast<unsigned char>(c))) { ++residues; continue; }
            if (c == '.') continue;
            if (column >= n)
                IMP_THROW("get_a3m_hits: row " << r << " has more match columns than the query",
                          ValueException);
            if (c != '-') {
                h.aligned[static_cast<std::size_t>(column)] = c;
                ++residues;
                if (first < 0) first = column;
                last = column;
            }
            ++column;
        }
        if (column != n)
            IMP_THROW("get_a3m_hits: row " << r << " has " << column << " match columns, the query "
                      << n, ValueException);
        if (first < 0) continue;
        // identity over the aligned span, insertions and deletions included
        int identical = 0, columns = 0;
        column = 0;
        for (char c : row) {
            if (c == '.') continue;
            const bool insertion = std::islower(static_cast<unsigned char>(c)) != 0;
            const bool inside = column > first && column <= last;
            if (insertion) {
                if (inside) ++columns;
                continue;
            }
            if (column >= first && column <= last) {
                ++columns;
                identical += c != '-' && c == query[static_cast<std::size_t>(column)];
            }
            ++column;
        }
        h.identity = columns > 0 ? double(identical) / columns : 0.0;
        h.query_start = first;
        h.query_end = last + 1;
        h.target_start = 0;
        h.target_end = residues;
        h.target_length = residues;
        // MMseqs2's header fields, when present
        std::istringstream fields(records[r].first);
        std::vector<std::string> f;
        for (std::string w; fields >> w;) f.push_back(w);
        h.identifier = f.empty() ? std::string() : f[0];
        if (f.size() >= 10) {
            char* end = nullptr;
            const double e = std::strtod(f[3].c_str(), &end);
            if (end != f[3].c_str()) h.evalue = e;
            h.score = std::atoi(f[1].c_str());
            h.target_length = std::max(std::atoi(f[9].c_str()), residues);
        }
        out.push_back(h);
    }
    return out;
}

IMPBFF_END_NAMESPACE
