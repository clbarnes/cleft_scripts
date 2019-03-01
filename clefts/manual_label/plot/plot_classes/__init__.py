from .area_histogram import AreaHistogramPlot
from .count_vs_area import CountVsAreaPlot
from .count_vs_avg_area import CountVsAvgAreaPlot
from .leftright_bias import LeftRightBiasPlot
from .frac_vs_area import FracVsAreaPlot
from .excit_inhib2 import ExcitationInhibitionPlot
from .heatmaps import ContactNumberHeatMap, SynapticAreaHeatMap, NormalisedDiffHeatMap, DendriticFractionHeatMap
from .compare_area_violin import CompareAreaViolinPlot
from .area_by_depth import DepthVsAreaPlot

# __all__ = [
#     "AreaHistogramPlot",
#     "CountVsAreaPlot",
#     "CountVsAreaPlot",
#     "LeftRightBiasPlot",
#     "FracVsAreaPlot",
#     "ExcitationInhibitionPlot",
#     "ContactNumberHeatMap",
#     "SynapticAreaHeatMap",
#     "NormalisedDiffHeatMap",
#     "DendriticFractionHeatMap",
#     "CompareAreaViolinPlot",
#     "DepthVsAreaPlot",
#     "circuit_plot_classes",
#     "combined_plot_classes"
# ]

circuit_plot_classes = [
    LeftRightBiasPlot,
    CountVsAreaPlot,
    FracVsAreaPlot,
    AreaHistogramPlot,
    ContactNumberHeatMap,
    SynapticAreaHeatMap,
    NormalisedDiffHeatMap,
    DendriticFractionHeatMap
]

combined_plot_classes = [
    CompareAreaViolinPlot,
    CountVsAvgAreaPlot,
    DepthVsAreaPlot,
    ExcitationInhibitionPlot,
]
