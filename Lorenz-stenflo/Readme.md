# Usage

You can execute the entire pipeline—from running hidden variants and symbolic refinement to final evaluation—by simply running:
```bash
run bash.sh
```

# Note on Matlab 
The script includes an automated step to compute evaluation metrics via MATLAB.

If `matlab` is available in your PATH: The script will automatically handle the metric calculations.
If `matlab` cannot be called from the terminal: Please manually run the following commands within the MATLAB Environment after the shell script finishes:
```matlab
metric
metric_adaptive
```