from typing import Optional, Union
from pathlib import Path
import subprocess

PathLike = Union[str, Path]


class HHBlits:
    """
    Args:
        database (str): UniClust database
        mact (float, optional): (0, 1] posterior prob threshold for MAC realignment
            controlling greediness at alignment ends: 0:global >0.1:local (default=0.35)
        maxfilt (int, optional): max number of hits allowed to pass 2nd prefilter
            (default=20000)
        neffmax (float, optional): (1,20] skip further search iterations when diversity
            Neff of query MSA becomes larger than neffmax (default=20.0)
        cpu (int, optional): number of CPUs to use (for shared memory SMPs) (default=2)
        all_seqs (bool, optional): show all sequences in result MSA; do not filter
            result MSA (default=False)
        realign_max (int, optional): realign max. <int> hits (default=500)
        maxmem (float, optional): [1,inf) limit memory for realignment (in GB)
            (default=3.0)
        n (int, optional): [1,8]   number of iterations (default=2)
        evalue (float, optional): E-value cutoff for inclusion in result alignment
            (default=0.001)
        verbose (bool, optional): whether to print information (default=False)
    """

    def __init__(
        self,
        database: str,
        mact: float = 0.35,
        maxfilt: int = 20000,
        neffmax: float = 20.0,
        cpu: int = 2,
        all_seqs: bool = False,
        realign_max: int = 500,
        maxmem: float = 3,
        n: int = 2,
        verbose: bool = False,
        evalue: float = 0.001,
    ):
        command = [
            "hhblits",
            f"-d {database}",
            f"-mact {mact}",
            f"-maxfilt {maxfilt}",
            f"-neffmax {neffmax}",
            f"-cpu {cpu}",
            f"-realign_max {realign_max}",
            f"-maxmem {maxmem}",
            f"-n {n}",
            "-o /dev/null",
            f"-v {0 if not verbose else 2}",
        ]

        if all_seqs:
            command.append("-all")

        if not verbose:
            command.append("-o /dev/null")

        self.command = " ".join(command)
        self._evalue = evalue

    def run(self, input_file: PathLike, out_prefix: Optional[PathLike] = None) -> None:
        if out_prefix is None:
            out_prefix = input_file
        out_path = Path(out_prefix).with_suffix(".a3m")
        extra = f"-i {input_file} -oa3m {out_path} -e {self.evalue}"
        command = self.command.split() + extra.split()
        print(" ".join(command))
        result = subprocess.run(command)
        result.check_returncode()

    @property
    def evalue(self) -> float:
        return self._evalue

    @evalue.setter
    def evalue(self, val):
        self._evalue = val


class HHFilter:
    """
    Args:
        verbose (bool, optional): Verbose mode (default=False)
        seqid (int, optional): [0,100]  maximum pairwise sequence identity (%)
            (default=90)
        diff (int, optional): [0,inf)  filter MSA by selecting most diverse set of
            sequences, keeping at least this many seqs in each MSA block of length 50
            (default=0)
        cov (int, optional): [0,100]  minimum coverage with query (%) (default=0)
        qid (int, optional): [0,100] minimum sequence identity with query (%)
            (default=0)
        qsc (float, optional): [0,100] minimum score per column with query
            (default=-20.0)
        neff (float, optional): [1,inf]  target diversity of alignment (default=off)

        M (str, optional): One of <'a2m', 'first', [0, 100]>
            * a2m: use A2M/A3M (default): upper case = Match; lower case = Insert;
                    '-' = Delete; '.' = gaps aligned to inserts (may be omitted)
            * first: use FASTA: columns with residue in 1st sequence are match states
            * [0,100] use FASTA: columns with fewer than X% gaps are match states
        maxseq (int, optional): max number of input rows (def=65535)
        maxres (int, optional): max number of HMM columns (def=20001)
    """

    def __init__(
        self,
        verbose: bool = False,
        seqid: int = 90,
        diff: int = 0,
        cov: int = 0,
        qid: int = 0,
        qsc: float = -20,
        neff: float = -1,
        M: str = "a2m",
        maxseq: int = 65535,
        maxres: int = 20001,
    ):
        command = [
            "hhfilter",
            f"-v {0 if not verbose else 2}",
            f"-id {seqid}",
            f"-diff {diff}",
            f"-cov {cov}",
            f"-qid {qid}",
            f"-qsc {qsc}",
            f"-neff {neff}",
            f"-M {M}",
            f"-maxseq {maxseq}",
            f"-maxres {maxres}",
        ]

        self.command = " ".join(command)

    def run(
        self,
        input_file: PathLike,
        output_file: PathLike,
        append_file: Optional[PathLike] = None,
    ) -> None:
        extra = f"-i {input_file} -o {output_file}"
        if append_file is not None:
            extra = extra + f" -a {append_file}"
        command = self.command.split() + extra.split()
        result = subprocess.run(command)
        result.check_returncode()


def count_sequences(fasta_file: PathLike) -> int:
    num_seqs = subprocess.check_output(f'grep "^>" -c {fasta_file}', shell=True)
    return int(num_seqs)


def remove_descriptions(fasta_file: PathLike) -> None:
    input_file = Path(fasta_file)
    output_file = input_file.with_suffix(".a3m.bk")
    command = f"cat {input_file} | " + r"awk '{print $1}' > " + str(output_file)
    subprocess.run(command, shell=True)
    output_file.rename(input_file)


def make_a3m(input_file: str, database: str, keep_intermediates: bool = False) -> None:
    hhblits = HHBlits(
        database,
        mact=0.35,
        maxfilt=100000000,
        neffmax=20,
        cpu=20,
        all_seqs=True,
        realign_max=10000000,
        maxmem=64,
        n=4,
        verbose=False,
    )

    hhfilter_id90cov75 = HHFilter(seqid=90, cov=75, verbose=False)
    hhfilter_id90cov50 = HHFilter(seqid=90, cov=50, verbose=False)

    output_file = Path(input_file).with_suffix(".a3m")
    if output_file.exists():
        raise FileExistsError(f"{output_file} already exists!")

    prev_a3m = Path(input_file)
    intermediates = []

    for evalue in [1e-80, 1e-60, 1e-40, 1e-20, 1e-10, 1e-8, 1e-6, 1e-4, 1e-3, 1e-1]:
        # HHblits at particular evalue
        hhblits.evalue = evalue
        out_path = Path(input_file).with_name(f".{Path(input_file).stem}.{evalue}.a3m")
        if not out_path.exists():
            hhblits.run(prev_a3m, out_path)
        intermediates.append(out_path)

        # HHFilter id90, cov75
        id90cov75_path = Path(input_file).with_name(
            f".{Path(input_file).stem}.{evalue}.id90cov75.a3m"
        )
        intermediates.append(id90cov75_path)
        if not id90cov75_path.exists():
            hhfilter_id90cov75.run(out_path, id90cov75_path)
        if count_sequences(id90cov75_path) > 2000:
            id90cov75_path.rename(output_file)
            break

        # HHFilter id90, cov50
        id90cov50_path = Path(input_file).with_name(
            f".{Path(input_file).stem}.{evalue}.id90cov50.a3m"
        )
        intermediates.append(id90cov50_path)
        if not id90cov50_path.exists():
            hhfilter_id90cov50.run(out_path, id90cov50_path)
        if count_sequences(id90cov50_path) > 5000:
            id90cov50_path.rename(output_file)
            break

        prev_a3m = id90cov50_path

    else:
        id90cov50_path.rename(output_file)

    remove_descriptions(output_file)

    if not keep_intermediates:
        for intermediate in intermediates:
            intermediate.unlink()
