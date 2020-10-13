from typing import Union, List, Tuple

from Bio import SeqIO
from collections import OrderedDict, namedtuple
from pathlib import Path
import numpy as np
import string

alphabet = "ARNDCQEGHILKMFPSTWYV"
a2n = {a: n for n, a in enumerate(alphabet)}
FamilyData = namedtuple("FamilyData", ["msa", "contacts"])
IUPAC_CODES = OrderedDict(
    [
        ("Ala", "A"),
        ("Asx", "B"),
        ("Cys", "C"),
        ("Asp", "D"),
        ("Glu", "E"),
        ("Phe", "F"),
        ("Gly", "G"),
        ("His", "H"),
        ("Ile", "I"),
        ("Lys", "K"),
        ("Leu", "L"),
        ("Met", "M"),
        ("Asn", "N"),
        ("Pro", "P"),
        ("Gln", "Q"),
        ("Arg", "R"),
        ("Ser", "S"),
        ("Thr", "T"),
        ("Sec", "U"),
        ("Val", "V"),
        ("Trp", "W"),
        ("Xaa", "X"),
        ("Tyr", "Y"),
        ("Glx", "Z"),
    ]
)
three2one = {three.upper(): one for three, one in IUPAC_CODES.items()}


def one_hot(x, cat=None):
    """Onehot encodes a sequence of ints."""
    if cat is None:
        cat = np.max(x) + 1
    oh = np.concatenate((np.eye(cat), np.zeros([1, cat])))
    return oh[x]


def load_npz_data(pdb_npz_file: Union[Path, str], c_beta_cutoff: int = 8):
    """Loader for npz file containing both contacts and MSA."""
    pdb_npz_file = Path(pdb_npz_file)
    if not pdb_npz_file.exists():
        raise FileNotFoundError(pdb_npz_file)

    # pull the msa directly from the loaded npz
    fam_data = np.load(pdb_npz_file)
    msa = fam_data["msa"]

    ref = msa[0]
    len_ref = len(ref)
    print("N over L:\t{}".format(len(msa) / len_ref))
    print("Length of reference:\t{}".format(len_ref))
    # convert gaps (and pads) to 0s, one hot the rest
    msa = one_hot(msa, cat=20)

    # distance matrix
    dist = fam_data["dist6d"]
    nat_contacts = dist * ((dist > 0) & (dist < c_beta_cutoff))
    assert nat_contacts.shape[0] == len(ref), "Contacts different length than MSA."
    return FamilyData(msa, nat_contacts)


def parse_fasta(
    filename: Union[str, Path], a3m: bool = False
) -> Tuple[List[str], List[str]]:
    if a3m:
        rm_lc = str.maketrans(dict.fromkeys(string.ascii_lowercase))

        def process_record(record: SeqIO.SeqRecord):
            return record.description, str(record.seq).translate(rm_lc)

    else:

        def process_record(record: SeqIO.SeqRecord):
            return record.description, str(record.seq)

    with open(filename) as f:
        records = SeqIO.parse(f, "fasta")
        records = map(process_record, records)
        records = zip(*records)
        headers, sequences = tuple(records)
    return headers, sequences


def get_seqref(x: str) -> Tuple[List[int], List[int], List[int]]:
    # input: string
    # output
    #   -seq: unaligned sequence (remove gaps, lower to uppercase, numeric(A->0, R->1...))
    #   -ref: reference describing how each sequence aligns to the first (reference sequence)
    n, seq, ref, aligned_seq = 0, [], [], []
    for aa in x:
        if aa != "-":
            seq.append(a2n.get(aa.upper(), -1))
            if aa.islower():
                ref.append(-1)
                n -= 1
            else:
                ref.append(n)
                aligned_seq.append(seq[-1])
        else:
            aligned_seq.append(-1)
        n += 1
    return np.array(seq), np.array(ref), np.array(aligned_seq)


def load_a3m_msa(filename):
    """
    Given A3M file (from hhblits)
    return MSA (aligned), MS (unaligned) and ALN (alignment)
    """

    names, seqs = parse_fasta(filename)

    reference = seqs[0]
    # get the multiple sequence alignment
    max_len = 0
    ms, aln, msa = [], [], []
    for seq in seqs:
        seq_, ref_, aligned_seq_ = get_seqref(seq)
        max_len = max(max_len, len(seq_))
        ms.append(seq_)
        msa.append(aligned_seq_)
        aln.append(ref_)

    # pad each unaligned-sequence and alignment to same length
    for n in range(len(ms)):
        pad = max_len - len(ms[n])
        ms[n] = np.pad(ms[n], [0, pad], constant_values=-1)
        aln[n] = np.pad(aln[n], [0, pad], constant_values=-1)

    return one_hot(msa), one_hot(ms), one_hot(aln), reference


def parse_cf(filename, cutoff=0.001, sequence=None):
    # contact Y,1     Y,2     0.006281        MET     ARG
    n, cons = 0, []
    with open(filename, "r") as f:
        for line in f:
            line = line.rstrip()
            if line[:7] == "contact":
                _, _, i, _, j, p, _, _ = line.replace(",", " ").split()
                i, j, p = int(i), int(j), float(p)
                if i > n:
                    n = i
                if j > n:
                    n = j
            cons.append([i - 1, j - 1, p])
            if line.startswith("SEQUENCE") and sequence is not None:
                seq = line.split()[1:]
                seq = "".join(three2one[code] for code in seq)
                start = seq.index(sequence)
                end = start + len(sequence)
                break
        else:
            start = 0
            end = n
    cm = np.zeros([n, n])
    for i, j, p in cons:
        cm[i, j] = p
    contacts = cm + cm.T
    contacts = contacts[start:end, start:end]
    return contacts
