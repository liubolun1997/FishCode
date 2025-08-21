import pandas as pd
import numpy as np

def build_stream_with_all_data(df, attack_types,
                               min_normal=200, max_normal=1500,
                               min_attack=20, max_attack=200,
                               seed=42):
    # Construct a validation set: normal blocks are interspersed with attack blocks,
    # all data is used, and no repeated sampling is done
    np.random.seed(seed)
    blocks = []

    normal_df = df[df['type'] == 'normal'].copy()
    attack_dfs = {t: df[df['type'] == t].copy() for t in attack_types}

    normal_idx = 0

    while normal_idx < len(normal_df):
        n_len = np.random.randint(min_normal, max_normal+1)
        n_len = min(n_len, len(normal_df) - normal_idx)
        normal_block = normal_df.iloc[normal_idx: normal_idx + n_len]
        blocks.append(normal_block)
        normal_idx += n_len

        # Randomly decide whether to insert an attack block or not
        available_attacks = [t for t in attack_types if len(attack_dfs[t]) > 0]
        if len(available_attacks) == 0:
            # All attack data has been exhausted
            continue
        # 70% chance to insert an attack block
        if np.random.rand() < 0.7:
            atk_type = np.random.choice(available_attacks)
            atk_df = attack_dfs[atk_type]
            atk_len = np.random.randint(min_attack, max_attack+1)
            atk_len = min(atk_len, len(atk_df))
            atk_block = atk_df.sample(n=atk_len, replace=False, random_state=seed)
            blocks.append(atk_block)

            # Remove used data
            attack_dfs[atk_type] = atk_df.drop(atk_block.index)

    # After the loop ends, all the remaining normal data and attack data are stitched together
    if normal_idx < len(normal_df):
        blocks.append(normal_df.iloc[normal_idx:])

    for t in attack_types:
        if len(attack_dfs[t]) > 0:
            blocks.append(attack_dfs[t])

    # Stitch all the blocks
    stream_df = pd.concat(blocks, ignore_index=True)
    return stream_df

if __name__ == "__main__":
    input_csv = "./csvdataset/validation2.csv"
    output_csv = "./csvdataset/val_stream2.csv"

    df = pd.read_csv(input_csv)

    attack_types = ['dos','backdoor','scanning','xss','ddos',
                    'password','injection','ransomware','mitm']

    # Generate a validation set
    val_stream = build_stream_with_all_data(df, attack_types,
                                            min_normal=200,
                                            max_normal=1500,
                                            min_attack=20,
                                            max_attack=200,
                                            seed=42)

    # save .csv
    val_stream.to_csv(output_csv, index=False)
    print(f"The validation set has been generated: {output_csv}, {len(val_stream)} rows")
    print("The number of samples in each category：")
    print(val_stream['type'].value_counts())
