## irsel

**Information Retrieval axiom SELection**

irsel is an experimental algorithm to perform [axiom selection for large theory reasoning](http://doi.org/10.1007/978-3-642-22438-6_23) ([slides](slides.pdf)).
It represents axioms as TF-IDF vectors, optionally applies [LSI](https://www.cs.bham.ac.uk/~pxt/IDA/lsa_ind.pdf) and repeatedly selects the best axioms in terms of cosine similarity to the conjecture.

### Setup

To set up irsel, choose one of the following options.

- **Using Vagrant**: With [VirtualBox](https://www.virtualbox.org/wiki/Downloads) and [Vagrant](https://www.vagrantup.com/downloads.html) installed, run `vagrant up`, `vagrant ssh` and `cd /vagrant`.
- **Manual setup on Linux**: To set up on a Linux system with Python 3, install some additional dependencies: `python3 -m pip install gavel gensim`
- **Manual setup on Windows**: On a Windows machine, install [WSL](https://docs.microsoft.com/de-de/windows/wsl/install-win10) and, from wsl.exe, the Python dependencies.

### Usage

Run irsel with `./irsel.sh <args>` or, in cmd.exe on Windows, `irsel <args>` (`-h` for help).

To prove theorems from [TPTP](http://www.tptp.org/), [download TPTP](http://www.tptp.org/TPTP/Distribution/TPTP-v7.3.0.tgz) to the `tptp` folder and run `./irsel.sh tptp/Problems/<some problem>`.

### Evaluation

Run the following to try proving the given example problems with identity, sine and irsel selectors (`-a`), irsel variations (`-e`), detailed comparisons (`-c`), and one minute timeout for EProver (`-t 60`). **Note that running this can take up to six hours.**

```
./irsel.sh examples/* -aceqt 60 > evaluation_quiet.txt 2>&1 # for a short summary
./irsel.sh examples/* -acevt 60 > evaluation_verbose.txt 2>&1 # creates a detailed report including selected axioms
```