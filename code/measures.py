#!/usr/bin/python3.7

import argparse
import itertools as it
import math
from collections import Counter, defaultdict
from functools import lru_cache
from time import perf_counter

import numpy as np
from scipy.special import entr, xlog1py

import dill
from bitarray import bitarray
from generate_quantifiers import QuantifierGenerator
from numba import jit


class MeasureCalculator:
    MONOTONICITY = 'monotonicity'
    QUANTITATIVITY = 'quantitativity'
    CONSERVATIVITY = 'conservativity'
    COMPLEXITY = 'complexity'

    LEFT = 'l'
    RIGHT = 'r'

    RU = 'ru'
    RD = 'rd'
    LU = 'lu'
    LD = 'ld'

    def __init__(self, upper, precomp_dir, verbose):
        self.start_t = perf_counter()
        self.upper = upper
        self.precomp_dir = precomp_dir
        self.verbose = verbose
        self.precomputed = None

        # Precompute information that does not depend on specific quantifiers
        self.precompute()

    def compute(self, q_true, label):
        measure_funs = {
            MeasureCalculator.MONOTONICITY: self.compute_monotonicity,
            MeasureCalculator.QUANTITATIVITY: self.compute_quantitativity,
            MeasureCalculator.CONSERVATIVITY: self.compute_conservativity,
            MeasureCalculator.COMPLEXITY: self.compute_complexity
        }

        results = {}

        for measure in measure_funs:
            results[measure] = measure_funs[measure](q_true)
            if self.verbose:
                print(f'{measure} computation for quantifier {label} finished after'
                      f' {(perf_counter() - self.start_t):.3f}s')

        return results

    def compute_monotonicity(self, q_true):
        """
        Computes each of the four types of degrees of monotonicity for quantifiers
        (represented as binary strings).

        Parameters
        ----------
        q_true : array-like[int]
            The quantifier's bitstring.

        Returns
        -------
        dict (string : float)
            Dict contains the four degrees of monotonicity, indexed
            by the monotonicity type name.
        """
        corresp_models = self.precomputed[MeasureCalculator.MONOTONICITY]

        monotonicities = {}

        if all(q_true) or not any(q_true):
            for mon_type in corresp_models:
                monotonicities[mon_type] = 1.0
        else:
            p_q_true = sum(q_true) / len(q_true)

            # Entropy of the quantifier's truth-value r.v.
            # xlog1py takes care of zero probabilities. Abs is taken to account
            # for negative zero
            H_q_true = abs((entr(p_q_true) -
                            xlog1py(1 - p_q_true, -p_q_true)) / np.log(2))

            for mon_type in corresp_models:
                # Build truth-value string for the monotonicity r.v. (w.r.t.
                # mon_type)
                q_true_mon = q_true.copy()

                for i in range(len(q_true_mon)):
                    if not q_true_mon[i]:
                        corresp_indices = corresp_models[mon_type][i]
                        for j in corresp_indices:
                            if q_true_mon[j]:
                                q_true_mon[i] = 1
                                break

                # Probability that the monotonicity r.v. is true
                p_q_true_mon = sum(q_true_mon) / len(q_true_mon)

                # Condition original truth-value r.v. on the monotonicity r.v.
                q_true_cond_mon = [q_true[i]
                                   for i in range(len(q_true))
                                   if q_true_mon[i]]

                if len(q_true_cond_mon) == 0:
                    H_q_true_cond_mon = 0
                else:
                    q_true_cond_mon = bitarray(q_true_cond_mon)

                    # Probability of original truth-value r.v. being true
                    # conditioned on the monotonicity r.v. being true
                    p_q_true_cond_mon = sum(
                        q_true_cond_mon) / len(q_true_cond_mon)

                    H_q_true_cond_mon = abs(p_q_true_mon *
                                            (entr(p_q_true_cond_mon) -
                                             xlog1py(1 - p_q_true_cond_mon,
                                                     -p_q_true_cond_mon)) /
                                            np.log(2))

                monotonicities[mon_type] = 1 - (H_q_true_cond_mon / H_q_true)

        return monotonicities

    def compute_quantitativity(self, q_true):
        """
        Computes the degree of quantitativity for quantifiers (represented as
        binary strings).

        Parameters
        ----------
        q_true : array-like[int]
            The quantifier's bitstring.

        Returns
        -------
        float
            Degree of quantitativity of the quantifier.
        """
        rv_preimage, rv_probs = self.precomputed[MeasureCalculator.QUANTITATIVITY]

        if all(q_true) or not any(q_true):
            quantitativity = 1.0
        else:
            # Probability that the quantifier is true
            p_q_true = sum(q_true) / len(q_true)

            # Entropy of the quantifier's truth-value r.v.
            # xlog1py takes care of zero probabilities. Abs is taken to account
            # for negative zero
            H_q_true = abs((entr(p_q_true) -
                            xlog1py(1 - p_q_true, -p_q_true)) / np.log(2))

            # Compute the conditional entropy of the truth-value r.v.
            # conditioned on the quantitativity r.v.
            H_cond = 0

            # For every value the quantitativity r.v. takes:
            for quan_tuple in rv_probs:
                # Probability of the value quan_tuple being output by the
                # quantitativity r.v.
                p = rv_probs[quan_tuple]
                q_cond_quan_indices = rv_preimage[quan_tuple]

                # Condition original quantifier string on the output of the
                # r.v.
                q_cond_vals = bitarray(len(q_cond_quan_indices))

                for new_i, original_i in enumerate(q_cond_quan_indices):
                    q_cond_vals[new_i] = q_true[original_i]

                # Probability of the quantifier being true conditioned on the
                # given output of the quantitativity r.v.
                p_q_cond_vals = sum(q_cond_vals) / len(q_cond_vals)

                H_cond += abs(p * (
                    entr(p_q_cond_vals) -
                    xlog1py(1 - p_q_cond_vals, -p_q_cond_vals) / np.log(2)))

            quantitativity = 1 - (H_cond / H_q_true)

        return quantitativity

    def compute_conservativity(self, q_true):
        """
        Computes the degrees of conservativity for quantifiers (represented as
        binary strings).

        Parameters
        ----------
        q_true : array-like[int]
            The quantifier's bitstring.

        Returns
        -------
        dict of (string : float)
            Mapping sending conservativity type to corresponding degree of
            conservativity of the quantifier's bitstring.
        """
        rv_preimage, rv_probs = self.precomputed[MeasureCalculator.CONSERVATIVITY]

        conservativities = {}

        if all(q_true) or not any(q_true):
            for cons_type in rv_preimage:
                conservativities[cons_type] = 1.0
        else:
            # Probability that the quantifier is true
            p_q_true = sum(q_true) / len(q_true)

            # Entropy of the quantifier's truth-value r.v.
            # xlog1py takes care of zero probabilities. Abs is taken to account
            # for negative zero
            H_q_true = abs((entr(p_q_true) -
                            xlog1py(1 - p_q_true, -p_q_true)) / np.log(2))

            for cons_type in rv_preimage:
                # Compute the conditional entropy of the truth-value r.v.
                # conditioned on the conservativity r.v.
                H_cond = 0

                # For every value the conservativity r.v. takes:
                for cons_tuple in rv_probs[cons_type]:
                    # Probability of the value cons_tuple being output by the
                    # conservativity r.v.
                    p = rv_probs[cons_type][cons_tuple]
                    q_cond_conserv_indices = rv_preimage[cons_type][cons_tuple]

                    # Condition original quantifier string on the output of the
                    # r.v.
                    q_cond_vals = bitarray(len(q_cond_conserv_indices))

                    for new_i, original_i in enumerate(q_cond_conserv_indices):
                        q_cond_vals[new_i] = q_true[original_i]

                    # Probability of the quantifier being true conditioned on the
                    # given output of the conservativity r.v.
                    p_q_cond_vals = sum(q_cond_vals) / len(q_cond_vals)

                    H_cond += abs(p * (
                        entr(p_q_cond_vals) -
                        xlog1py(1 - p_q_cond_vals, -p_q_cond_vals) / np.log(2)))

                conservativities[cons_type] = 1 - (H_cond / H_q_true)

        return conservativities

    def compute_complexity(self, q_true):
        """
        Approximates the (average) Kolmogorov complexity of binary strings
        representing quantifiers' truth values.

        We average over the approximate Kolmogorov complexities of the different
        permutations of the string, one for each way to order the models (ternary
        strings) based on different lexicographic orderings of 0, 1, and 2. These
        individual Kolmogorov complexities of (permutations of) strings are computed
        by averaging over the Lempel-Ziv complexities of both the string and its
        reverse, and then multiplying this number by the logarithm of the length of the string.

        Parameters
        ----------
        q_true : array-like[int]
            The quantifier's bitstrings.

        Returns
        -------
        float
            Approximate Kolmogorov complexity of the quantifier's bitstring.
        """
        permutations = self.precomputed[MeasureCalculator.COMPLEXITY]

        complexity = 0

        for p in permutations:
            q_true_permuted = p(q_true)
            complexity += compute_lempel_ziv(q_true_permuted)
            complexity += compute_lempel_ziv(
                q_true_permuted[::-1])

        complexity /= 2 * len(permutations)
        complexity *= np.log2(len(q_true))

        return complexity

    def precompute(self):
        """
        Computes necessary data required for measure computation, and potentially
        stores it.
        """
        helper_funs = {MeasureCalculator.MONOTONICITY: self.precompute_monotonicity,
                       MeasureCalculator.QUANTITATIVITY: self.precompute_quantitativity,
                       MeasureCalculator.CONSERVATIVITY: self.precompute_conservativity,
                       MeasureCalculator.COMPLEXITY: self.precompute_complexity}

        precomputed = {}

        for measure in helper_funs:
            try:
                with open(f'{self.precomp_dir}/{measure}_helper.pickle', 'rb') as f:
                    precomputed[measure] = dill.load(f)
            except IOError:
                precomputed[measure] = helper_funs[measure]()
                if self.precomp_dir is not None:
                    with open(f'{self.precomp_dir}/{measure}_helper.pickle', 'wb') as f:
                        dill.dump(precomputed[measure], f)

        if self.verbose:
            print('Precomputation finished after'
                  f' {(perf_counter() - self.start_t):.3f}s')

        self.precomputed = precomputed

    def precompute_monotonicity(self):
        """
        Computes relevant information about the random variables involved in the
        computation of quantifiers' degrees of monotonicity.

        Returns
        -------
        corresp_models : dict of (string : dict of (int : set[int]))
            Indexed by the type of monotonicity, contains mapping sending model
            indices to the set of model indices whose truth value is relevant for
            the monotonicity r.v. of that type.
        """
        # Check functions used in computing specific monotonicities
        check_funs = {MeasureCalculator.RU: MeasureCalculator.check_right_upward,
                      MeasureCalculator.RD: MeasureCalculator.check_right_downward,
                      MeasureCalculator.LU: MeasureCalculator.check_left_upward,
                      MeasureCalculator.LD: MeasureCalculator.check_left_downward}

        base_idx_i = 0

        total_model_count = sum(3 ** n for n in range(1, self.upper + 1))

        # For each of the four types of monotonicity, associate with each model
        # index the set of model indices that are a sub/supermodel of that original
        # model according to the monotonicity type.
        corresp_models = {check_name: defaultdict(
            set) for check_name in check_funs}

        for model_size_1 in range(1, self.upper + 1):
            for i, model_i in enumerate(it.product(range(3), repeat=model_size_1)):
                base_idx_j = 0
                for model_size_2 in range(1, self.upper + 1):
                    for j, model_j in enumerate(it.product(range(3),
                                                           repeat=model_size_2)):
                        for check_name in check_funs:
                            check = check_funs[check_name]
                            # If model_j is a sub/super model of model_i according
                            # to check:
                            if check(model_i, model_j):
                                corresp_models[check_name][base_idx_i +
                                                           i].add(base_idx_j + j)
                    base_idx_j += 3 ** model_size_2
                if self.verbose:
                    print('Computing sub/super models of model'
                          f' {base_idx_i + i + 1}/{total_model_count} for'
                          f' {MeasureCalculator.MONOTONICITY} precomputation done after'
                          f' {(perf_counter() - self.start_t):.3f}s')
            base_idx_i += 3 ** model_size_1

        if self.verbose:
            print(f'{MeasureCalculator.MONOTONICITY} precomputation finished after'
                  f' {(perf_counter() - self.start_t):.3f}s')

        return corresp_models

    @staticmethod
    def check_right_upward(model_i, model_j):
        """
        Determines if a model i is a right-superset of a model j.

        Parameters
        ----------
        model_i, model_j : array-like[int]
            Ternary strings representing the models.

        Returns
        -------
        bool
        """
        A_i, B_i = QuantifierGenerator.build_lists(model_i)
        A_j, B_j = QuantifierGenerator.build_lists(model_j)

        return set(B_j) <= set(B_i) and A_i == A_j

    @staticmethod
    def check_right_downward(model_i, model_j):
        """
        Determines if a model i is a right-subset of a model j.

        Parameters
        ----------
        model_i, model_j : array-like[int]
            Ternary strings representing the models.

        Returns
        -------
        bool
        """
        return check_right_upward(model_j, model_i)

    @staticmethod
    def check_left_upward(model_i, model_j):
        """
        Determines if a model i is a left-superset of a model j.

        Parameters
        ----------
        model_i, model_j : array-like[int]
            Ternary strings representing the models.

        Returns
        -------
        bool
        """
        A_i, B_i = QuantifierGenerator.build_lists(model_i)
        A_j, B_j = QuantifierGenerator.build_lists(model_j)

        return set(A_j) <= set(A_i) and B_i == B_j

    @staticmethod
    def check_left_downward(model_i, model_j):
        """
        Determines if a model i is a left-subset of a model j.

        Parameters
        ----------
        model_i, model_j : array-like[int]
            Ternary strings representing the models.

        Returns
        -------
        bool
        """
        return check_left_upward(model_j, model_i)

    def precompute_quantitativity(self):
        """
        Computes relevant information about the random variables involved in the
        computation of quantifiers' degree of quantitativity.

        Returns
        -------
        rv_preimage : dict of (tuple[int, int, int] : set[int])
            Mapping sending tuples (x, y, z) to the set of indices of models (A, B)
            such that x, y, and z are the sizes of the relevant zones of
            (A, B) (i.e. the quantitativity random variable's pre-image of (x, y, z)).
        rv_probs : dict of (tuple[int, int, int] : float)
            Mapping sending tuple (x, y, z) to the probability the quantitativity
            random variable takes the value (x, y, z).
        """
        rv_preimage = defaultdict(set)
        rv_probs = {}

        # Counter keeping track of how often some tuple appears in the image of the
        # r.v.
        rv_counter = Counter()

        base_idx = 0

        # For every model...
        for model_size in range(1, self.upper + 1):
            for i, model in enumerate(it.product(range(3), repeat=model_size)):
                c = Counter()
                c.update({n: 0 for n in range(3)})
                c.update(model)
                # ... compute cardinalities of its three zones
                vals = tuple(c.values())

                rv_preimage[vals].add(base_idx + i)
                rv_counter.update((vals,))

            base_idx += 3 ** model_size

        rv_counter_sum = sum(rv_counter.values())

        for vals in rv_counter:
            rv_probs[vals] = rv_counter[vals] / rv_counter_sum

        if self.verbose:
            print(f'{MeasureCalculator.QUANTITATIVITY} precomputation finished after'
                  f' {(perf_counter() - self.start_t):.3f}s')

        return rv_preimage, rv_probs

    def precompute_conservativity(self):
        """
        Computes relevant information about the random variables involved in the
        computation of quantifiers' degree of conservativity.

        Returns
        -------
        rv_preimage : dict of (string : dict of (tuple[int, tuple, tuple] : set[int]))
            Mapping sending conservativity types to dicts, sending
            tuple (n, X, Y) to the set of indices of
            models (A, B) such that (X, Y) is the conservative restriction of (A, B)
            for the given type (i.e. the conservativity random variable's
            pre-image of (n, X, Y)).
        rv_probs : dict of (string : dict of (tuple[int, tuple, tuple] : float))
            Mapping sending conservativity types to dicts, sending
            tuple (n, X, Y) to the probability the conservativity
            random variable of the given type takes the value (n, X, Y).
        """
        rv_preimage = {MeasureCalculator.LEFT: defaultdict(set),
                       MeasureCalculator.RIGHT: defaultdict(set)}
        rv_probs = {MeasureCalculator.LEFT: {}, MeasureCalculator.RIGHT: {}}

        # Counter keeping track of how often some tuple appears in the image of the
        # r.v.
        rv_counter = {MeasureCalculator.LEFT: Counter(),
                      MeasureCalculator.RIGHT: Counter()}

        base_idx = 0

        # For every model...
        for model_size in range(1, self.upper + 1):
            for i, model in enumerate(it.product(range(3), repeat=model_size)):
                A, B = QuantifierGenerator.build_lists(model)

                # ...compute its conservative restrictions
                cons_tuple_l = (tuple(A), tuple(
                    sorted(set(A).intersection(B))))
                cons_tuple_r = (
                    tuple(sorted(set(A).intersection(B))), tuple(B))

                rv_preimage[MeasureCalculator.LEFT][cons_tuple_l].add(
                    base_idx + i)
                rv_preimage[MeasureCalculator.RIGHT][cons_tuple_r].add(
                    base_idx + i)
                rv_counter[MeasureCalculator.LEFT].update((cons_tuple_l,))
                rv_counter[MeasureCalculator.RIGHT].update((cons_tuple_r,))

            base_idx += 3 ** model_size

        rv_counter_sum_l = sum(rv_counter[MeasureCalculator.LEFT].values())
        rv_counter_sum_r = sum(rv_counter[MeasureCalculator.RIGHT].values())

        for cons_tuple in rv_counter[MeasureCalculator.LEFT]:
            rv_probs[MeasureCalculator.LEFT][cons_tuple] = (
                rv_counter[MeasureCalculator.LEFT][cons_tuple] /
                rv_counter_sum_l
            )

        for cons_tuple in rv_counter[MeasureCalculator.RIGHT]:
            rv_probs[MeasureCalculator.RIGHT][cons_tuple] = (
                rv_counter[MeasureCalculator.RIGHT][cons_tuple] /
                rv_counter_sum_r
            )

        if self.verbose:
            print(f'{MeasureCalculator.CONSERVATIVITY} precomputation finished after'
                  f' {(perf_counter() - self.start_t):.3f}s')

        return rv_preimage, rv_probs

    def precompute_complexity(self):
        """
        Construct functions permuting quantifiers' bitstrings according to
        different lexicographic orderings of ternary strings.

        Returns
        -------
        list of (array-like[int] -> list[int])
            The permutations.
        """
        # Since we compute the LZ complexity of a quantifier by taking the average
        # over the LZ complexity of that quantifier's bit string and that of its
        # reverse bitstring, we consider only 3!/2 lexicographic orderings,
        # instead of 3!
        lex_orders = [[-1, 0, 1, 2], [-1, 0, 2, 1], [-1, 1, 0, 2]]
        permutations = []
        models = []

        for model_size in range(1, self.upper + 1):
            # Ensure that all model strings are of equal length
            padding = (self.upper - model_size) * [-1]
            for model in it.product(range(3), repeat=model_size):
                models.append(padding + list(model))

        def permute_list(order):
            def permuter(b):
                return [b[i] for i in order]
            return permuter

        for order in lex_orders:
            permutation = sorted(range(len(models)),
                                 key=lambda i: [order.index(zone)
                                                for zone in models[i]])
            permutations.append(permute_list(permutation))

        if self.verbose:
            print(f'{MeasureCalculator.COMPLEXITY} precomputation finished after'
                  f' {(perf_counter() - self.start_t):.3f}s')

        return permutations


@jit
def compute_lempel_ziv(binary_sequence):
    """
    Computes the Lempel-Ziv complexity of a binary string.

    Parameters
    ----------
    binary_sequence : array-like[int]
        The input binary string.

    Returns
    -------
    int
        The Lempel-Ziv complexity of the string.

    Notes
    ------
    Source: https://naereen.github.io/Lempel-Ziv_Complexity/
    """
    u, v, w = 0, 1, 1
    v_max = 1
    length = len(binary_sequence)
    complexity = 1
    while True:
        if binary_sequence[u + v - 1] == binary_sequence[w + v - 1]:
            v += 1
            if w + v >= length:
                complexity += 1
                break
        else:
            if v > v_max:
                v_max = v
            u += 1
            if u == w:
                complexity += 1
                w += v_max
                if w > length:
                    break
                else:
                    u = 0
                    v = 1
                    v_max = 1
            else:
                v = 1
    return complexity


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-u', '--upper', type=int, required=True,
                        help='Model size upper bound')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Show progress and timing')
    parser.add_argument('-e', '--expr', type=str, required=True,
                        help='Path to file containing quantifier expressions')
    parser.add_argument('-b', '--bit', type=str, required=True,
                        help='Path to binary file containing quantifier'
                        ' bitstrings')
    parser.add_argument('-o', '--output', type=str, required=True,
                        help='Path final CSV file containing quantifier'
                        ' expressions and measures should be stored in')
    parser.add_argument('-p', '--precomp', type=str, default=None,
                        help='If given, path to directory in which pickle'
                        ' files containing precomputed data making later runs'
                        ' more efficient should be stored in')

    args = parser.parse_args()

    upper = args.upper
    verbose = args.verbose
    expr_file = args.expr
    bit_file = args.bit
    out_file = args.output
    precomp_dir = args.precomp

    mc = MeasureCalculator(upper, precomp_dir, verbose)

    total_models = sum(3**n for n in range(1, upper + 1))
    chunk_size = math.ceil(total_models / 8)

    with open(expr_file, 'r') as f_expr:
        with open(bit_file, 'rb') as f_bit:
            with open(out_file, 'w') as f_out:
                f_out.write(
                    'expr;mon_ru;mon_lu;mon_rd;mon_ld;quant;cons_l;cons_r;compl\n')

                expr = f_expr.readline()
                i = 1

                while expr:
                    expr = expr.strip()

                    byte_string = f_bit.read(chunk_size)
                    bitstring = bitarray()
                    bitstring.frombytes(byte_string)
                    bitstring = bitstring[:total_models]

                    results = mc.compute(bitstring, i)

                    monotonicity = results[MeasureCalculator.MONOTONICITY]
                    quantitativity = results[MeasureCalculator.QUANTITATIVITY]
                    conservativity = results[MeasureCalculator.CONSERVATIVITY]
                    complexity = results[MeasureCalculator.COMPLEXITY]

                    mon_ru = monotonicity[MeasureCalculator.RU]
                    mon_rd = monotonicity[MeasureCalculator.RD]
                    mon_lu = monotonicity[MeasureCalculator.LU]
                    mon_ld = monotonicity[MeasureCalculator.LD]

                    cons_l = conservativity[MeasureCalculator.LEFT]
                    cons_r = conservativity[MeasureCalculator.RIGHT]

                    f_out.write(f'{expr};')
                    f_out.write(f'{mon_ru};{mon_lu};{mon_rd};{mon_ld};')
                    f_out.write(f'{quantitativity};')
                    f_out.write(f'{cons_l};{cons_r};')
                    f_out.write(f'{complexity}\n')

                    expr = f_expr.readline()
                    i += 1


if __name__ == '__main__':
    main()
