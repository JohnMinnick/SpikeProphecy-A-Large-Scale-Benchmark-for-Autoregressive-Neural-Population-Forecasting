"""
Master script: generate ALL paper figures with unified style.

Usage:
    python -m scripts.figures.generate_all            # all figures
    python -m scripts.figures.generate_all --main      # main body only
    python -m scripts.figures.generate_all --appendix  # appendix only
    python -m scripts.figures.generate_all --fig 2     # specific figure

Output: docs/neurips_ed/figures/ (PNG + PDF at 300 DPI)
"""

import argparse
import sys
import time

from figures.style import apply_style


def main():
    """Parse arguments and generate requested figures."""
    parser = argparse.ArgumentParser(description='Generate SpikeProphecy paper figures')
    parser.add_argument('--main', action='store_true', help='Main body figures only')
    parser.add_argument('--appendix', action='store_true', help='Appendix figures only')
    parser.add_argument('--fig', type=int, help='Generate specific figure (1, 2, or 3)')
    args = parser.parse_args()

    # Apply unified style
    apply_style()

    t0 = time.time()
    print('=' * 60)
    print('SpikeProphecy — Unified Figure Generator')
    print('=' * 60)

    # Determine which figures to generate
    do_main = not args.appendix
    do_appendix = not args.main

    if args.fig:
        do_main = args.fig in [1, 2, 3]
        do_appendix = False

    # --- Main body figures ---
    if do_main:
        if not args.fig or args.fig == 1:
            from figures.figure1_hero_v4 import generate
            generate()

        if not args.fig or args.fig == 2:
            from figures.figure2_eval import generate
            generate()

        if not args.fig or args.fig == 3:
            from figures.figure3_findings import generate
            generate()

    # --- Appendix figures ---
    if do_appendix:
        from figures.figure_appendix import generate_all_appendix
        generate_all_appendix()

    elapsed = time.time() - t0
    print(f'\n{"=" * 60}')
    print(f'All figures generated in {elapsed:.1f}s')
    print('=' * 60)


if __name__ == '__main__':
    main()
