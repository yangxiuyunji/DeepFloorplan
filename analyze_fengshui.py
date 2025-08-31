"""Command line utility to analyze floorplan fengshui.

Run either *Luo Shu* missing-corner detection or *BaZhai* eight-star
analysis depending on ``--mode``:

.. code-block:: bash

    python analyze_fengshui.py path/to/floorplan.json --mode luoshu
    python analyze_fengshui.py path/to/floorplan.json --mode bazhai
"""
from fengshui.__main__ import main

if __name__ == "__main__":
    main()
