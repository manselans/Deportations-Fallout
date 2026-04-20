"""
Create figures and tables from the main text and Appendix.
"""


import deportations_fallout.results.tables as tbs
import deportations_fallout.results.figures as fgs

from deportations_fallout.config import PATHS, setup_matplotlib
paths = PATHS
setup_matplotlib()


def run_analysis():
    """
    Generate figures and tables from the main text.
    """

    # ---- Tables

    print("Generating tables", end="...", flush=True)

    print("1", end="...", flush=True)
    tbl = tbs.table_1()
    tbl.to_html(paths.tables/"tab1.html")

    print("2", end="...", flush=True)
    tbl = tbs.table_2()
    tbl.to_html(paths.tables/"tab2.html", escape=False)

    print("3")
    tbl = tbs.table_3()
    tbl.to_html(paths.tables/"tab3.html")

    # ---- Figures

    print("Generating figures", end="...", flush=True)

    print("1", end="...", flush=True)
    fig, _ = fgs.figure_1()
    fig.savefig(paths.figures/"fig1.tif", pil_kwargs={"compression": "tiff_lzw"})

    print("2", end="...", flush=True)
    fig, _ = fgs.figure_2()
    fig.savefig(paths.figures/"fig2.tif", pil_kwargs={"compression": "tiff_lzw"})

    print("3", end="...", flush=True)
    fig, _ = fgs.figure_3()
    fig.savefig(paths.figures/"fig3.tif", pil_kwargs={"compression": "tiff_lzw"})

    print("4", end="...", flush=True)
    fig, _ = fgs.figure_4()
    fig.savefig(paths.figures/"fig4.tif", pil_kwargs={"compression": "tiff_lzw"})

    print("5", end="...", flush=True)
    fig, _ = fgs.figure_5()
    fig.savefig(paths.figures/"fig5.tif", pil_kwargs={"compression": "tiff_lzw"})

    print("6", end="...", flush=True)
    fig, _ = fgs.figure_6()
    fig.savefig(paths.figures/"fig6.tif", pil_kwargs={"compression": "tiff_lzw"})

    print("7", end="...", flush=True)
    fig, _ = fgs.figure_7()
    fig.savefig(paths.figures/"fig7.tif", pil_kwargs={"compression": "tiff_lzw"})

    print("8", end="...", flush=True)
    fig, _ = fgs.figure_8()
    fig.savefig(paths.figures/"fig8.tif", pil_kwargs={"compression": "tiff_lzw"})

    print("9")
    fig, _ = fgs.figure_9()
    fig.savefig(paths.figures/"fig9.tif", pil_kwargs={"compression": "tiff_lzw"})

    # ---- Appendix

    print("Generating appendix tables", end="...", flush=True)

    print("A1", end="...", flush=True)
    tbl = tbs.table_a1()
    tbl.to_html(paths.tables/"tabA1.html")

    print("B1", end="...", flush=True)
    tbl, mccrary = tbs.table_b1()
    mccrary.to_stata(paths.temp / "mccrary.dta", write_index=False)
    tbl.to_html(paths.tables/"tabB1.html")

    print("B2", end="...", flush=True)
    tbl = tbs.table_b2()
    tbl.to_html(paths.tables/"tabB2.html", escape=False)

    print("B3", end="...", flush=True)
    tbl = tbs.table_b3()
    tbl.to_html(paths.tables/"tabB3.html")

    print("C1")
    tbl = tbs.table_c1()
    tbl.to_html(paths.tables/"tabC1.html")

    print("Generating appendix figures", end="...", flush=True)

    print("A1", end="...", flush=True)
    fig, _ = fgs.figure_a1()
    fig.savefig(paths.figures/"figA1.tif", pil_kwargs={"compression": "tiff_lzw"})

    print("B1", end="...", flush=True)
    fig, _ = fgs.figure_b1()
    fig.savefig(paths.figures/"figB1.tif", pil_kwargs={"compression": "tiff_lzw"})

    print("B2", end="...", flush=True)
    fig, _ = fgs.figure_b2()
    fig.savefig(paths.figures/"figB2.tif", pil_kwargs={"compression": "tiff_lzw"})

    print("C1", end="...", flush=True)
    fig, _ = fgs.figure_c1()
    fig.savefig(paths.figures/"figC1.tif", pil_kwargs={"compression": "tiff_lzw"})

    print("C2")
    fig, _ = fgs.figure_c2()
    fig.savefig(paths.figures/"figC2.tif", pil_kwargs={"compression": "tiff_lzw"})

    print("Done.")


if __name__ == "__main__":
    run_analysis()
