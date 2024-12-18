# 2024 AAML final project - group 3

## How to Compile

1. Navigate to the directory `./proj/AAML_final_proj`.
2. Compile the project using the following command:
   ```bash
   make prog EXTRA_LITEX_ARGS="--cpu-variant=perf+cfu"
   ```
3. Load the compiled program using:
    ```bash
    make load
    ```
4. After completing the above steps, run the evaluation script by executing:
    ```bash
    python eval_script.py --port /dev/ttyUSB1
    ```
    Replace `/dev/ttyUSB1` with the appropriate port for your device if necessary.