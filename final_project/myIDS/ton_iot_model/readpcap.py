import pandas as pd
from pathlib import Path

numeric_columns = [
    'src_port', 'dst_port', 'duration', 'src_bytes', 'dst_bytes', 'missed_bytes',
    'src_pkts', 'src_ip_bytes', 'dst_pkts', 'dst_ip_bytes', 'dns_qclass',
    'dns_qtype', 'dns_rcode', 'http_request_body_len', 'http_response_body_len', 'http_status_code'
]

categorical_columns = [
    'proto', 'service', 'conn_state', 'dns_query', 'dns_AA', 'dns_RD', 'dns_RA',
    'dns_rejected', 'ssl_version', 'ssl_cipher', 'ssl_resumed', 'ssl_established',
    'ssl_subject', 'ssl_issuer', 'http_trans_depth', 'http_method', 'http_version',
    'http_orig_mime_types', 'http_resp_mime_types', 'weird_name', 'weird_addl', 'weird_notice'
]

extra_categorical = ['src_ip', 'dst_ip', 'http_uri', 'http_user_agent']
for c in extra_categorical:
    if c not in categorical_columns:
        categorical_columns.insert(0, c)

# Adjust the column order for the final CSV
model_features = [
    'src_ip', 'src_port', 'dst_ip', 'dst_port', 'proto', 'service', 'duration',
    'src_bytes', 'dst_bytes', 'conn_state', 'missed_bytes', 'src_pkts', 'src_ip_bytes',
    'dst_pkts', 'dst_ip_bytes', 'dns_query', 'dns_qclass', 'dns_qtype', 'dns_rcode',
    'dns_AA', 'dns_RD', 'dns_RA', 'dns_rejected', 'ssl_version', 'ssl_cipher',
    'ssl_resumed', 'ssl_established', 'ssl_subject', 'ssl_issuer', 'http_trans_depth',
    'http_method', 'http_uri', 'http_version', 'http_request_body_len', 'http_response_body_len',
    'http_status_code', 'http_user_agent', 'http_orig_mime_types', 'http_resp_mime_types',
    'weird_name', 'weird_addl', 'weird_notice', 'label', 'type'
]


# A function that reads the Zeek log
def read_zeek_log(path):
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()

    fields_line = None
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            if line.startswith("#fields"):
                fields_line = line.strip()
                break
    if not fields_line:
        return pd.DataFrame()

    columns = fields_line.split("\t")[1:]
    try:
        df = pd.read_csv(path, sep="\t", comment="#", header=None, names=columns, encoding='utf-8', dtype=str)
    except pd.errors.EmptyDataError:
        df = pd.DataFrame(columns=columns)
    return df


# Functions that read and merge a catalog of individual datasets
def readnewcsv(path, label, typ, rows_num=0):
    base_path = Path(path)

    # read .log data
    conn_df = read_zeek_log(base_path / "conn.log")
    dns_df = read_zeek_log(base_path / "dns.log")
    ssl_df = read_zeek_log(base_path / "ssl.log")
    http_df = read_zeek_log(base_path / "http.log")
    weird_df = read_zeek_log(base_path / "weird.log")

    if conn_df.empty:
        # Create a uid-based DF (take the uid from the first non-empty DF)
        merged_df = None
        for df in (dns_df, ssl_df, http_df, weird_df):
            if not df.empty and 'uid' in df.columns:
                merged_df = df[['uid']].copy()
                break
        if merged_df is None:
            merged_df = pd.DataFrame(columns=['uid'])
    else:
        merged_df = conn_df.copy()

    # Merge by uid (only if the corresponding DF is non-empty and contains a uid)
    def safe_merge(left, right, **kwargs):
        if right is None or right.empty or 'uid' not in right.columns:
            return left
        return left.merge(right, on='uid', how='left', **kwargs)

    merged_df = safe_merge(merged_df, dns_df, suffixes=("", "_dns"))
    merged_df = safe_merge(merged_df, ssl_df, suffixes=("", "_ssl"))
    merged_df = safe_merge(merged_df, http_df, suffixes=("", "_http"))
    merged_df = safe_merge(merged_df, weird_df, suffixes=("", "_weird"))

    column_mapping = {
        "id.orig_h": "src_ip",
        "id.orig_p": "src_port",
        "id.resp_h": "dst_ip",
        "id.resp_p": "dst_port",
        "orig_bytes": "src_bytes",
        "resp_bytes": "dst_bytes",
        "orig_pkts": "src_pkts",
        "resp_pkts": "dst_pkts",
        "orig_ip_bytes": "src_ip_bytes",
        "resp_ip_bytes": "dst_ip_bytes",
        "query": "dns_query",
        "qclass": "dns_qclass",
        "qtype": "dns_qtype",
        "rcode": "dns_rcode",
        "AA": "dns_AA",
        "RD": "dns_RD",
        "RA": "dns_RA",
        "rejected": "dns_rejected",
        "version": "ssl_version",
        "cipher": "ssl_cipher",
        "resumed": "ssl_resumed",
        "established": "ssl_established",
        "subject": "ssl_subject",
        "issuer": "ssl_issuer",
        "trans_depth": "http_trans_depth",
        "method": "http_method",
        "uri": "http_uri",
        "version_http": "http_version",
        "request_body_len": "http_request_body_len",
        "response_body_len": "http_response_body_len",
        "status_code": "http_status_code",
        "user_agent": "http_user_agent",
        "orig_mime_types": "http_orig_mime_types",
        "resp_mime_types": "http_resp_mime_types",
        "id": "weird_name",
        "addl": "weird_addl",
        "notice": "weird_notice"
    }

    merged_df = merged_df.rename(columns=column_mapping)

    # Add the label and type columns
    merged_df['label'] = label
    merged_df['type'] = typ

    for col in numeric_columns:
        if col not in merged_df.columns:
            merged_df[col] = 0
        else:
            merged_df[col] = pd.to_numeric(merged_df[col], errors='coerce').fillna(0)

    for col in categorical_columns:
        if col not in merged_df.columns:
            merged_df[col] = '-'
        else:
            merged_df[col] = merged_df[col].fillna('-').astype(str)

    for col in extra_categorical:
        if col not in merged_df.columns:
            merged_df[col] = '-'
        else:
            merged_df[col] = merged_df[col].fillna('-').astype(str)

    # Now take the subset in final column order
    final_df = merged_df.reindex(columns=model_features, fill_value='-' if model_features else None)

    # Set the numeric column type to float explicitly
    for col in numeric_columns:
        final_df[col] = pd.to_numeric(final_df[col], errors='coerce').fillna(0)

    if rows_num and isinstance(rows_num, int) and rows_num > 0:
        final_df = final_df.head(rows_num)

    return final_df


# read .csv data
normal_df1 = readnewcsv(
    r"D:/毕设/TON_IoT datasets/Raw_datasets/network_data/Network_dataset_Bro/normal_Bro/normal_1",
    label=0, typ='normal'
)
normal_df2 = readnewcsv(
    r"D:/毕设/TON_IoT datasets/Raw_datasets/network_data/Network_dataset_Bro/normal_Bro/normal_2",
    label=0, typ='normal'
)
normal_df3 = readnewcsv(
    r"D:/毕设/TON_IoT datasets/Raw_datasets/network_data/Network_dataset_Bro/normal_Bro/normal_3",
    label=0, typ='normal'
)
normal_df4 = readnewcsv(
    r"D:/毕设/TON_IoT datasets/Raw_datasets/network_data/Network_dataset_Bro/normal_Bro/normal_4",
    label=0, typ='normal'
)
normal_df5 = readnewcsv(
    r"D:/毕设/TON_IoT datasets/Raw_datasets/network_data/Network_dataset_Bro/normal_Bro/normal_5",
    label=0, typ='normal'
)
normal_df6 = readnewcsv(
    r"D:/毕设/TON_IoT datasets/Raw_datasets/network_data/Network_dataset_Bro/normal_Bro/normal_6",
    label=0, typ='normal'
)

backdoor_df = readnewcsv(
    r"D:/毕设/TON_IoT datasets/Raw_datasets/network_data/Network_dataset_Bro/normal_attack_Bro/normal_backdoor",
    label=1, typ='backdoor', rows_num=20000
)
ddos_df = readnewcsv(
    r"D:/毕设/TON_IoT datasets/Raw_datasets/network_data/Network_dataset_Bro/normal_attack_Bro/normal_DDoS/normal_DDoS_1",
    label=1, typ='ddos', rows_num=20000
)
dos_df = readnewcsv(
    r"D:/毕设/TON_IoT datasets/Raw_datasets/network_data/Network_dataset_Bro/normal_attack_Bro/normal_DoS/norma1_DoS_1",
    label=1, typ='dos', rows_num=20000
)
password_df = readnewcsv(
    r"D:/毕设/TON_IoT datasets/Raw_datasets/network_data/Network_dataset_Bro/normal_attack_Bro/normal_password/password_normal_1",
    label=1, typ='password', rows_num=20000
)
injection_df = readnewcsv(
    r"D:/毕设/TON_IoT datasets/Raw_datasets/network_data/Network_dataset_Bro/normal_attack_Bro/normal_injection/injection_normal2",
    label=1, typ='injection', rows_num=20000
)
ransomware_df = readnewcsv(
    r"D:/毕设/TON_IoT datasets/Raw_datasets/network_data/Network_dataset_Bro/normal_attack_Bro/normal_runsomware",
    label=1, typ='ransomware', rows_num=20000
)
scanning_df = readnewcsv(
    r"D:/毕设/TON_IoT datasets/Raw_datasets/network_data/Network_dataset_Bro/normal_attack_Bro/normal_scanning/normal_scanning_1",
    label=1, typ='scanning', rows_num=20000
)
xss_df = readnewcsv(
    r"D:/毕设/TON_IoT datasets/Raw_datasets/network_data/Network_dataset_Bro/normal_attack_Bro/normal_XSS/normal_XSS1",
    label=1, typ='xss', rows_num=20000
)

# Merge and output the file name used by the model (the same as the file name read in your model)
all_df = pd.concat([normal_df1, normal_df2, normal_df3, normal_df4, normal_df5, normal_df6,
                    backdoor_df, injection_df, ddos_df, password_df, dos_df, ransomware_df, scanning_df, xss_df],
                   ignore_index=True)

output_path = "iot_features_mapped.csv"
all_df.to_csv(output_path, index=False)
print(f"{output_path}，rows：{len(all_df)}")
