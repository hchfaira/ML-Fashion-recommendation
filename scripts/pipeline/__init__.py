"""
Pipeline Testing Package
========================

Modular components for testing the full outfit recommendation pipeline.

Modules:
- results_storage: Save garments, combinations, and generate graphs
- pipeline_tester: Main testing class for wardrobe analysis
- report_generator: Generate formatted reports and visualizations
- graph_generator: Generate visual charts and graphs
"""

from .results_storage import ResultsStorage
from .pipeline_tester import FullPipelineTester
from .report_generator import ReportGenerator, run_demo
from . import graph_generator

__all__ = [
    "ResultsStorage",
    "FullPipelineTester", 
    "ReportGenerator",
    "run_demo",
    "graph_generator",
]
