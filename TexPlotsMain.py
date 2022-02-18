import LatexPlots.Metrics as Metrics
import LatexPlots.Vectors as Vectors
import LatexPlots.Summary as Summary


if __name__ == '__main__':
    # Metrics.MetricPlots(index = 2, Number = 5, Number_of_orbits = 30, first = True, ALL = False, width = 8.0, height = 6.0)
    # Vectors.VectorPlots(index = 1, Number = 2, Number_of_orbits = 30, first = True, ALL = False, width = 8.0, height = 6.0)
    Summary.SummaryPlots(index = 2, Number = 30, Number_of_orbits = 30, first = True, ALL = True, width = 8.0, height = 6.0)