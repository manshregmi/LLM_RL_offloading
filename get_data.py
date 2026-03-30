import csv

def read_prefill_dec_values(csv_filepath):
    """
    Reads the CSV file row by row and outputs prefill_dec_iprof_ms values
    with seq_len starting at 100 and increasing.
    """
    with open(csv_filepath, 'r') as file:
        reader = csv.DictReader(file)
        
        seq_counter = 0
        for row in reader:
            prefill_dec_value = float(row['step_iprof_ms'])
            # input_mbit = float(row['input_mbit'])
            # kv_cache_mbit = float(row['kv_cache_mbit'])
            # cross_kv_mbit = float(row['cross_kv_mbit'])
            


            print(f"({seq_counter},0): {prefill_dec_value},")
            # print(f"({seq_counter},0): {(input_mbit+kv_cache_mbit)/8},")

            seq_counter += 1

if __name__ == "__main__":
    csv_filepath = "/Users/Manish/Downloads/llama_autoregressive_kv_cache_jetpack51_100.csv"
    read_prefill_dec_values(csv_filepath)