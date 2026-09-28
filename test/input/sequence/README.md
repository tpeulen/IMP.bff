# Sequence test data

`rbp_60x120.fasta`: 60 sequences x 120 columns cut from the ribose-binding
protein alignment `Test/inputs/OutMSA.MSA_filtered30` of FRETNet-Designer
(SMB-Lab, MIT licence): every second of the first 150 sequences, first 120
columns. The sequences are UniProt entries (CC BY 4.0); names are the UniProt
identifiers.

Reference outputs, produced by running the original programs on that file
as black boxes (their outputs are data; no code of theirs is in this
repository):

- `rate4site_bg.res`: `rate4site -s rbp_60x120.fasta` (Rate4Site 3.0.0, the
  Debian package, on x86_64 Linux): its defaults, branch lengths refitted
  under the gamma model.
- `rate4site_bn.res`: the same with `-bn`, the setting ConSurf runs.
- `dca_reference.dat`: FRETNet-Designer's mean-field DCA script
  (`Script_Dependancies/DCA.py`) with theta 0.2: every ninth pair of columns
  and the twenty with the largest direct information.

The A/B tests in `test/sequence/` compare `IMP.bff`'s independent
implementations against these.
