import csv
import os 
def read_prefill_dec_values(csv_filepath):
    """
    Reads the CSV file row by row and outputs prefill_dec_iprof_ms values
    with seq_len starting at 100 and increasing.
    """
    with open(csv_filepath, 'r') as file:
        reader = csv.DictReader(file)
        
        seq_counter = 150
        for row in reader:
            prefill_dec_value = float(row['step_total_ms'])
            # input_mbit = float(row['input_mbit'])
            # input_mbit = float(row['dec_input_mbit'])
            # kv_cache_mbit = float(row['self_kv_cache_mbit'])
            # cross_kv_mbit = float(row['cross_kv_mbit'])
            


            print(f"({seq_counter},0): {prefill_dec_value},")
            # print(f"({seq_counter},0): {(input_mbit+kv_cache_mbit+cross_kv_mbit)/8},")

            seq_counter += 1

if __name__ == "__main__":
    # csv_filepath = os.path.join("bart_summary_profiling_400.csv")
    csv_filepath = r"C:\Users\SIU856622975\Desktop\bart_summary_profiling_200_agx.csv"
    # csv_filepath = r"C:\Users\SIU856622975\Downloads\llama_autoregressive_kv_cache_jetpack51_100.csv"


    read_prefill_dec_values(csv_filepath)