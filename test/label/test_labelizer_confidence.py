"""pLDDT in the Labelizer and the network design: model_confidence rows from an
AlphaFold model's B-factors, unchanged combined scores, and pair costs that
make pairs with an unreliable site ineligible."""
import math
import os

import numpy as np
import pytest

import IMP
import IMP.bff as bff

AF = os.path.join(os.path.dirname(__file__), "..", "..", "examples", "structure", "MBP", "AF-P0AEX9-F1.pdb")


@pytest.fixture(scope="module")
def scored():
    IMP.set_log_level(IMP.SILENT)
    model = bff.labelizer_model_paper()
    plain = bff.labelizer_score_structure(AF, model, bff.LabelizerOptions())
    o = bff.LabelizerOptions()
    o.bfactor_is_confidence = True
    with_conf = bff.labelizer_score_structure(AF, model, o)
    return plain, with_conf


def test_confidence_rows_hold_the_plddt_and_leave_the_label_score_alone(scored):
    plain, with_conf = scored
    assert not bff.labelizer_confidence_by_key(plain)
    conf = bff.labelizer_confidence_by_key(with_conf)
    assert len(conf) == 396
    # the signal peptide (residues 1-26) is predicted with low confidence
    assert np.mean([conf[f"A{r}"] for r in range(1, 27)]) < 70
    assert np.mean([conf[f"A{r}"] for r in range(30, 370)]) > 90
    assert dict(bff.labelizer_combined_by_key(with_conf)) == dict(bff.labelizer_combined_by_key(plain))


def test_pair_costs_exclude_pairs_with_a_low_confidence_site():
    conf = [95.0, 40.0, 80.0, float("nan")]
    pairs = np.array([[0, 2], [0, 1], [2, 3], [2, 0]], dtype=np.int32)
    c = bff.probe_pair_confidence_costs(conf, pairs)
    assert c[0] == 0.0 and c[3] == 0.0
    assert math.isnan(c[1]) and math.isnan(c[2])
    term = bff.ProbePairCostTerm(list(c))
    assert term.get_is_eligible_pair(0) and not term.get_is_eligible_pair(1)
    with pytest.raises(bff.IndexException):
        bff.probe_pair_confidence_costs(conf, np.array([[0, 9]], dtype=np.int32))
