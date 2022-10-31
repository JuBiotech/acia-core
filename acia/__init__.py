"""Top-level package for AutomatedCellularImageAnalysis."""

__author__ = """Johannes Seiffarth"""
__email__ = "j.seiffarth@fz-juelich.de"
__version__ = "0.2.29"


from pint import UnitRegistry

ureg = UnitRegistry()
Q_ = ureg.Quantity
U_ = ureg.Unit
