PYTHON ?= python

install:
	$(PYTHON) -m pip install -r requirements.txt

run:
	$(PYTHON) scripts/run_end_to_end.py --config configs/edge_experiment.yaml

plot:
	$(PYTHON) src/visualization/plot_benchmarks.py --csv reports/benchmark_results.csv --output reports/figures/latency_plot.png

clean:
	rm -rf outputs reports/benchmark_results.csv reports/figures/latency_plot.png assets/demo.gif
