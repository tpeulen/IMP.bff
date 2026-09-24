/**
 *  \file IMP/bff/internal/NetworkDocument.h
 *  \brief The one door between a network and its msgpack bytes.
 *
 * msgpack is bff's native format for every neural-network document: the
 * `bff.neural_net` network itself, the action policy a model search reads
 * (the same document plus a `"temperature"`), and the documents that nest a
 * network as a map (`bff.hmm_surrogate`). float64 weights survive exactly,
 * the bytes are a third the size of the text, and there is no second
 * reader: every producer and consumer in bff goes through the functions
 * here. nlohmann::json is only the in-memory tree they decode into;
 * `mlpcore::model_from_json` / `model_to_json` map that tree to an MlpModel.
 *
 * Copyright 2007-2026 IMP Inventors. All rights reserved.
 */

#ifndef IMPBFF_INTERNAL_NETWORK_DOCUMENT_H
#define IMPBFF_INTERNAL_NETWORK_DOCUMENT_H

#include <IMP/bff/bff_config.h>
#include <IMP/bff/IMPCompatibility.h>
#include <IMP/bff/NeuralNet.h>
#include <IMP/bff/internal/json.h>
#include <IMP/bff/internal/MlpCore.h>

#include <cstdint>
#include <exception>
#include <string>
#include <vector>

IMPBFF_BEGIN_NAMESPACE
namespace internal {

//! Decode msgpack bytes to a map whose `"format"` is `format`.
/*! `who` prefixes the messages. Strict: trailing bytes are an error.
    \throws IMP::ValueException on empty, malformed or non-map input, or a
            missing or different `"format"`. */
inline nlohmann::json document_from_msgpack(const MsgpackBytes& bytes,
                                            const std::string& format,
                                            const std::string& who) {
  if (bytes.empty())
    IMP_THROW(who << ": the document is empty (expected msgpack bytes of a '"
                  << format << "' map)",
              IMP::ValueException);
  const nlohmann::json doc = nlohmann::json::from_msgpack(bytes, true, false);
  if (doc.is_discarded())
    IMP_THROW(who << ": the document is not valid msgpack (expected a '"
                  << format << "' map)",
              IMP::ValueException);
  if (!doc.is_object())
    IMP_THROW(who << ": the msgpack document is not a map (expected a '"
                  << format << "' map)",
              IMP::ValueException);
  const auto found = doc.find("format");
  if (found == doc.end() || !found->is_string())
    IMP_THROW(who << ": the document has no \"format\" (expected '" << format
                  << "')",
              IMP::ValueException);
  const std::string got = found->get<std::string>();
  if (got != format)
    IMP_THROW(who << ": unexpected format '" << got << "', expected '" << format
                  << "'",
              IMP::ValueException);
  return doc;
}

//! Encode a document tree as msgpack bytes.
inline MsgpackBytes document_to_msgpack(const nlohmann::json& doc) {
  const std::vector<std::uint8_t> bytes = nlohmann::json::to_msgpack(doc);
  return MsgpackBytes(bytes.begin(), bytes.end());
}

//! A `bff.neural_net` tree (already decoded) as a validated model.
inline MlpModel model_from_document(const nlohmann::json& doc,
                                    const std::string& who = "NeuralNet") {
  if (!doc.is_object() || !doc.contains("format") || !doc.at("format").is_string() ||
      doc.at("format").get<std::string>() != "bff.neural_net")
    IMP_THROW(who << ": the network is not a 'bff.neural_net' map", IMP::ValueException);
  try {
    MlpModel model = mlpcore::model_from_json(doc);
    model.validate();
    return model;
  } catch (const std::exception& e) {
    IMP_THROW(who << ": malformed 'bff.neural_net' document: " << e.what(),
              IMP::ValueException);
  }
}

//! Parse a `bff.neural_net` msgpack document; validates before returning.
inline MlpModel model_from_msgpack(const MsgpackBytes& bytes,
                                   const std::string& who = "NeuralNet") {
  return model_from_document(document_from_msgpack(bytes, "bff.neural_net", who), who);
}

//! Serialise a model as a `bff.neural_net` msgpack document.
inline MsgpackBytes model_to_msgpack(const MlpModel& model) {
  return document_to_msgpack(mlpcore::model_to_json<nlohmann::json>(model));
}

}  // namespace internal
IMPBFF_END_NAMESPACE

#endif  // IMPBFF_INTERNAL_NETWORK_DOCUMENT_H
