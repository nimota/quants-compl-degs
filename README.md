# quants-compl-degs
Code corresponding to a paper I wrote called "Quantifiers, complexity, and degrees of semantic universals: A large-scale analysis"

In order to reproduce the results from the paper:
- Execute `./generate_quantifiers.py -u 10 -d 6 -v -i data/cfg.txt -e data/quantifiers.txt -b data/quantifiers.bin`, renaming/creating all paths as needed
- Run `./measures.py -u 10 -v -e data/quantifiers.txt -b data/quantifiers.bin -o measures.csv -p data/precomp`
- Open up the notebook to get plots and statistics

Note that this code is **extremely slow**. Even on a modern PC, running generate_quantifiers with depth 6 takes hours, generating them up to depth 7 will take weeks, if not months (it has yet to finish running on my system), and will require gigabytes of RAM and storage (we have combinatorial explosion to thank for that). The precomputation step of measures.py can also take up to a week, while the measure computation itself requires between one and two days.
I can only imagine this code's performance on non-extensional quantifiers...

Feel free to send me an e-mail if you have a question/suggestion - address is in the paper.


