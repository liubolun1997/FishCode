import pandas as pd
import numpy as np

def build_attack_stream(df, attack_types,
                        min_block=50, max_block=300,
                        seed=42):
    # Construct an attack data stream: Randomly select block stitches from multiple attack types
    # Each category is used in chunks until all data is exhausted
    np.random.seed(seed)
    blocks = []

    # Split data by category
    attack_dfs = {t: df[df['category'] == t].copy() for t in attack_types}

    while any(len(atk_df) > 0 for atk_df in attack_dfs.values()):
        # Select a category with remaining data
        available_attacks = [t for t, atk_df in attack_dfs.items() if len(atk_df) > 0]
        atk_type = np.random.choice(available_attacks)

        atk_df = attack_dfs[atk_type]
        block_len = np.random.randint(min_block, max_block + 1)
        block_len = min(block_len, len(atk_df))

        # A random block is drawn
        atk_block = atk_df.sample(n=block_len, replace=False, random_state=seed)
        blocks.append(atk_block)

        # Remove used data
        attack_dfs[atk_type] = atk_df.drop(atk_block.index)

    # Stitch all the blocks
    stream_df = pd.concat(blocks, ignore_index=True)
    return stream_df


if __name__ == "__main__":
    # Input and output files
    input_csv = "./csvdataset/validation1.csv"
    output_csv = "./csvdataset/val_stream1.csv"

    df = pd.read_csv(input_csv)

    # The type of attack you currently hold
    attack_types = ['DDoS', 'DoS', 'Reconnaissance']

    # Generate a validation set
    val_stream = build_attack_stream(df, attack_types,
                                     min_block=50,
                                     max_block=300,
                                     seed=42)

    # save
    val_stream.to_csv(output_csv, index=False)
    print(f"The validation set has been generated: {output_csv}, {len(val_stream)} rows")
    print("The number of samples in each category:")
    print(val_stream['category'].value_counts())
