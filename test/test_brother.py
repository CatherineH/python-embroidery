from pattern_utils import csv_to_pattern

from brother import BrotherEmbroideryFile


def test_import_csv():
    input_file = 'test.csv'
    pattern = csv_to_pattern(input_file)
    bef = BrotherEmbroideryFile(input_file + ".pes")
    bef.write_pattern(pattern)