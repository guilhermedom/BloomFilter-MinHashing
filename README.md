# Bloom Filter MinHashing

Text anomaly detection algorithm using MinHash and Bloom Filter. MinHashing is used to find signatures representations for input documents. A Bloom Filter is used to create bins of similar signatures. Similar signatures are inserted in the same Bloom Filter bins due to their similar results for the hash functions used in the Bloom Filter. Additionally, the usage of a Bloom Filter speeds up the necessary comparisons between signatures.

At the end, document minhash signatures that occupy the most empty spaces in the Bloom Filter have larger anomaly scores since they are more isolated.

Input is a Bag of Words, Document Frequency x Term Frequency matrix. Output is a vector ranking of anomaly scores with anomalies in the top positions.

