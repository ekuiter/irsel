## irsel

**Information Retrieval axiom SELection**

irsel is an experimental algorithm to perform [axiom selection for large theory reasoning](http://doi.org/10.1007/978-3-642-22438-6_23).
It represents axioms as TF-IDF vectors, optionally applies [LSI](https://www.cs.bham.ac.uk/~pxt/IDA/lsa_ind.pdf) and repeatedly selects the best axioms in terms of cosine similarity to the conjecture.

### Setup

To set up irsel, choose one of the following options.

- **Using Vagrant**: With [VirtualBox](https://www.virtualbox.org/wiki/Downloads) and [Vagrant](https://www.vagrantup.com/downloads.html) installed, run `vagrant up`, `vagrant ssh` and `cd /vagrant`.
- **Manual setup on Linux**: To set up on a Linux system with Python 3, install some additional dependencies: `python3 -m pip install gavel gensim`
- **Manual Setup on Windows**: On a Windows machine, install [WSL-Ubuntu](https://docs.microsoft.com/de-de/windows/wsl/install-win10) and install the dependencies like above from wsl.exe.

### Usage

Run irsel with `./irsel.sh <args>` or `./irsel <args>` on Windows from cmd.exe (`-h` for help).

To prove theorems from [TPTP](http://www.tptp.org/), [download the TPTP](http://www.tptp.org/TPTP/Distribution/TPTP-v7.3.0.tgz) to the `tptp` folder and run `./irsel.sh tptp/Problems/<some problem>`.

### Evaluation

For example, run the following to try proving the given example problems with no selector (identity), sine and irsel (with 5 minute timeout for EProver):

```
./irsel.sh examples/* -s all -t 300 -q > results.txt 2>&1 # for a short summary like below
./irsel.sh examples/* -s all -t 300 -v -c > results.txt 2>&1 # creates a detailed report including selected axioms
```

On my machine, this produces the following `results.txt`. **Note that running this can take up to six hours.**

```
- Problem examples/problem1.p -
selector        proof steps     selection time  proof time      selection ratio
identity        15              0:00:00.000028  0:00:00.117501  100.0% (3 of 3)
sine            15              0:00:00.000235  0:00:00.125998  100.0% (3 of 3)
irsel           15              0:00:00.017264  0:00:00.113495  100.0% (3 of 3)

- Problem examples/big.p -
selector        proof steps     selection time  proof time      selection ratio
identity        31642           0:00:00.000064  0:00:07.428928  100.0% (10006 of 10006)
sine            -               0:00:00.529080  0:00:00.114061  0.03% (3 of 10006)
irsel           51              0:00:02.169561  0:00:00.127930  0.14% (14 of 10006)

- Problem examples/BIO002+1.p -
selector        proof steps     selection time  proof time      selection ratio
identity        -               0:00:00.000007  0:05:08.635971  100.0% (9161 of 9161)
sine            -               0:02:33.177620  0:00:00.244862  0.02% (2 of 9161)
irsel           -               0:00:52.071374  0:05:22.827146  0.26% (24 of 9161)

- Problem examples/NLP218+1.p -
selector        proof steps     selection time  proof time      selection ratio
identity        -               0:00:00.000055  0:05:09.116891  100.0% (71 of 71)
sine            -               0:00:00.022543  0:05:09.131126  66.2% (47 of 71)
irsel           -               0:00:00.494549  0:05:06.269325  57.75% (41 of 71)

- Problem examples/NLP261+1.p -
selector        proof steps     selection time  proof time      selection ratio
identity        -               0:00:00.000007  0:05:08.947741  100.0% (1026860 of 1026860)
sine            212             3:26:14.930589  0:00:00.421662  0.01% (72 of 1026860)
irsel           -               1:14:20.622899  0:00:00.441516  0.0% (10 of 1026860)

- Problem examples/NUM006+1.p -
selector        proof steps     selection time  proof time      selection ratio
identity        -               0:00:00.000015  0:05:22.861702  100.0% (18 of 18)
sine            -               0:00:00.023707  0:05:16.513577  100.0% (18 of 18)
irsel           -               0:00:00.370419  0:05:10.360900  83.33% (15 of 18)
```