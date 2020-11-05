from typing import Union, List, Tuple, Dict, Optional

from Bio import SeqIO
from biotite.structure.io.pdb import PDBFile
from scipy.spatial.distance import pdist, squareform
from pathlib import Path
import numpy as np
import string
from .vocab import FastaVocab

PathLike = Union[str, Path]


def one_hot(x, cat=None):
    """Onehot encodes a sequence of ints."""
    if cat is None:
        cat = np.max(x) + 1
    oh = np.concatenate((np.eye(cat), np.zeros([1, cat])))
    return oh[x]


def parse_fasta(
    filename: Union[str, Path],
    remove_insertions: bool = False,
    remove_gaps: bool = False,
) -> Tuple[List[str], List[str]]:

    translate_dict: Dict[str, Optional[str]] = {}
    if remove_insertions:
        translate_dict.update(dict.fromkeys(string.ascii_lowercase))
    else:
        translate_dict.update(dict(zip(string.ascii_lowercase, string.ascii_uppercase)))

    if remove_gaps:
        translate_dict["-"] = None

    translate_dict["."] = None
    translation = str.maketrans(translate_dict)

    def process_record(record: SeqIO.SeqRecord):
        return record.description, str(record.seq).translate(translation)

    records = SeqIO.parse(str(filename), "fasta")
    records = map(process_record, records)
    records = zip(*records)
    headers, sequences = tuple(records)
    return headers, sequences


def get_seqref(x: str) -> Tuple[List[int], List[int], List[int]]:
    # input: string
    # output
    #   -seq: unaligned sequence (remove gaps, lower to uppercase,
    #           numeric(A->0, R->1...))
    #   -ref: reference describing how each sequence aligns to the first
    #           (reference sequence)
    n, seq, ref, aligned_seq = 0, [], [], []
    for aa in x:
        if aa != "-":
            seq.append(FastaVocab.A2N.get(aa.upper(), -1))
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


def contacts_from_cf(filename: PathLike, cutoff=0.001, sequence=None) -> np.ndarray:
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
                seq = "".join(FastaVocab.THREE_LETTER[code] for code in seq)
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


def extend(a, b, c, L, A, D):
    """
    input:  3 coords (a,b,c), (L)ength, (A)ngle, and (D)ihedral
    output: 4th coord
    """

    def normalize(x):
        return x / np.linalg.norm(x, ord=2, axis=-1, keepdims=True)

    bc = normalize(b - c)
    n = normalize(np.cross(b - a, bc))
    m = [bc, np.cross(n, bc), n]
    d = [L * np.cos(A), L * np.sin(A) * np.cos(D), -L * np.sin(A) * np.sin(D)]
    return c + sum([m * d for m, d in zip(m, d)])


def contacts_from_pdb(
    filename: PathLike, distance_threshold: float = 8.0
) -> np.ndarray:
    pdbfile = PDBFile.read(str(filename))
    structure = pdbfile.get_structure()

    N = structure.coord[0, structure.atom_name == "N"]
    C = structure.coord[0, structure.atom_name == "C"]
    CA = structure.coord[0, structure.atom_name == "CA"]

    Cbeta = extend(C, N, CA, 1.522, 1.927, -2.143)
    distogram = squareform(pdist(Cbeta))
    return distogram < distance_threshold


def contacts_from_trrosetta(
    filename: PathLike,
    distance_threshold: float = 8.0,
):
    fam_data = np.load(filename)
    dist = fam_data["dist6d"]
    nat_contacts = dist * ((dist > 0) & (dist < distance_threshold))
    return nat_contacts


def read_contacts(filename: PathLike, **kwargs) -> np.ndarray:
    filename = Path(filename)
    if filename.suffix == ".cf":
        return contacts_from_cf(filename, **kwargs)
    elif filename.suffix == ".pdb":
        return contacts_from_pdb(filename, **kwargs)
    elif filename.suffix == ".npz":
        return contacts_from_trrosetta(filename, **kwargs)
    else:
        raise ValueError(
            f"Cannot read file of type {filename.suffix}, must be one of (.cf, .pdb)"
        )
